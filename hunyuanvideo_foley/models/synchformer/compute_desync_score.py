import argparse
import subprocess
from pathlib import Path

import torch
import torchaudio
import torchvision
from omegaconf import OmegaConf

import data_transforms
from .synchformer import Synchformer
from .data_transforms import make_class_grid, quantize_offset
from .utils import check_if_file_exists_else_download, which_ffmpeg


def prepare_inputs(batch, device):
    aud = batch["audio"].to(device)
    vid = batch["video"].to(device)

    return aud, vid


def get_test_transforms():
    ts = [
        data_transforms.EqualifyFromRight(),
        data_transforms.RGBSpatialCrop(input_size=224, is_random=False),
        data_transforms.TemporalCropAndOffset(
            crop_len_sec=5,
            max_off_sec=2,  # https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml
            max_wiggle_sec=0.0,
            do_offset=True,
            offset_type="grid",
            prob_oos="null",
            grid_size=21,
            segment_size_vframes=16,
            n_segments=14,
            step_size_seg=0.5,
            vfps=25,
        ),
        data_transforms.GenerateMultipleSegments(
            segment_size_vframes=16,
            n_segments=14,
            is_start_random=False,
            step_size_seg=0.5,
        ),
        data_transforms.RGBToHalfToZeroOne(),
        data_transforms.RGBNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # motionformer normalization
        data_transforms.AudioMelSpectrogram(
            sample_rate=16000,
            win_length=400,  # 25 ms * 16 kHz
            hop_length=160,  # 10 ms * 16 kHz
            n_fft=1024,  # 2^(ceil(log2(window_size * sampling_rate)))
            n_mels=128,  # as in AST
        ),
        data_transforms.AudioLog(),
        data_transforms.PadOrTruncate(max_spec_t=66),
        data_transforms.AudioNormalizeAST(mean=-4.2677393, std=4.5689974),  # AST, pre-trained on AudioSet
        data_transforms.PermuteStreams(
            einops_order_audio="S F T -> S 1 F T", einops_order_rgb="S T C H W -> S T C H W"  # same
        ),
    ]
    transforms = torchvision.transforms.Compose(ts)

    return transforms


def get_video_and_audio(path, get_meta=False, start_sec=0, end_sec=None):
    orig_path = path
    # (Tv, 3, H, W) [0, 255, uint8]; (Ca, Ta)
    rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, "sec", output_format="TCHW")
    assert meta["video_fps"], f"No video fps for {orig_path}"
    # (Ta) <- (Ca, Ta)
    audio = audio.mean(dim=0)
    # FIXME: this is legacy format of `meta` as it used to be loaded by VideoReader.
    meta = {
        "video": {"fps": [meta["video_fps"]]},
        "audio": {"framerate": [meta["audio_fps"]]},
    }
    return rgb, audio, meta


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != "", "Is ffmpeg installed? Check if the conda environment is activated."
    new_path = Path.cwd() / "vis" / f"{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f"{which_ffmpeg()}"
    # no info/error printing
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -y -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f" {new_path}"
    subprocess.call(cmd.split())
    cmd = f"{which_ffmpeg()}"
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -y -i {new_path}"
    cmd += f" -acodec pcm_s16le -ac 1"
    cmd += f' {new_path.replace(".mp4", ".wav")}'
    subprocess.call(cmd.split())
    return new_path


def decode_single_video_prediction(off_logits, grid, item):
    label = item["targets"]["offset_label"].item()
    print("Ground Truth offset (sec):", f"{label:.2f} ({quantize_offset(grid, label)[-1].item()})")
    print()
    print("Prediction Results:")
    off_probs = torch.softmax(off_logits, dim=-1)
    k = min(off_probs.shape[-1], 5)
    topk_logits, topk_preds = torch.topk(off_logits, k)
    # remove batch dimension
    assert len(topk_logits) == 1, "batch is larger than 1"
    topk_logits = topk_logits[0]
    topk_preds = topk_preds[0]
    off_logits = off_logits[0]
    off_probs = off_probs[0]
    for target_hat in topk_preds:
        print(f'p={off_probs[target_hat]:.4f} ({off_logits[target_hat]:.4f}), "{grid[target_hat]:.2f}" ({target_hat})')
    return off_probs


def main(args):
    vfps = 25
    afps = 16000
    in_size = 256
    # making the offset class grid similar to the one used in transforms,
    # refer to the used one: https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml
    max_off_sec = 2
    num_cls = 21

    # checking if the provided video has the correct frame rates
    print(f"Using video: {args.vid_path}")
    v, _, info = torchvision.io.read_video(args.vid_path, pts_unit="sec")
    _, H, W, _ = v.shape
    if info["video_fps"] != vfps or info["audio_fps"] != afps or min(H, W) != in_size:
        print(f'Reencoding. vfps: {info["video_fps"]} -> {vfps};', end=" ")
        print(f'afps: {info["audio_fps"]} -> {afps};', end=" ")
        print(f"{(H, W)} -> min(H, W)={in_size}")
        args.vid_path = reencode_video(args.vid_path, vfps, afps, in_size)
    else:
        print(f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={in_size}')

    device = torch.device(args.device)

    # load visual and audio streams
    # rgb: (Tv, 3, H, W) in [0, 225], audio: (Ta,) in [-1, 1]
    rgb, audio, meta = get_video_and_audio(args.vid_path, get_meta=True)

    # making an item (dict) to apply transformations
    # NOTE: here is how it works:
    # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
    # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
    # track by `args.offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
    # start by `offset_sec` earlier than the rgb track.
    # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (-2, +2) seconds (see `grid`)
    item = dict(
        video=rgb,
        audio=audio,
        meta=meta,
        path=args.vid_path,
        split="test",
        targets={
            "v_start_i_sec": args.v_start_i_sec,
            "offset_sec": args.offset_sec,
        },
    )

    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
    if not (min(grid) <= item["targets"]["offset_sec"] <= max(grid)):
        print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {grid}')

    # applying the test-time transform
    item = get_test_transforms()(item)

    # prepare inputs for inference
    batch = torch.utils.data.default_collate([item])
    aud, vid = prepare_inputs(batch, device)

    # TODO:
    # sanity check: we will take the input to the `model` and recontruct make a video from it.
    # Use this check to make sure the input makes sense (audio should be ok but shifted as you specified)
    # reconstruct_video_from_input(aud, vid, batch['meta'], args.vid_path, args.v_start_i_sec, args.offset_sec,
    #                              vfps, afps)

    # forward pass
    with torch.set_grad_enabled(False):
        with torch.autocast("cuda", enabled=True):
            _, logits = synchformer(vid, aud)

    # simply prints the results of the prediction
    decode_single_video_prediction(logits, grid, item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, help="In a format: xx-xx-xxTxx-xx-xx")
    parser.add_argument("--vid_path", required=True, help="A path to .mp4 video")
    parser.add_argument("--offset_sec", type=float, default=0.0)
    parser.add_argument("--v_start_i_sec", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    synchformer = Synchformer().cuda().eval()
    synchformer.load_state_dict(
        torch.load(
            os.environ.get("SYNCHFORMER_WEIGHTS", f"weights/synchformer.pth"),
            weights_only=True,
            map_location="cpu",
        )
    )

    main(args)
