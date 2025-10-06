import re
import argparse
import os

def get_last_match(pattern, text, cast_func):
    match=re.findall(pattern, text)

    return cast_func(match[-1]) if match else None




def main(args):
    # Read log file
    file_name = args.log_file
    with open(file_name, "r") as f:
        log_text = f.read()

    # Regex patterns
    pattern_time = re.compile(r"Sparse reconstruction took (\d+) sec")
    pattern_PSNR = re.compile(r"PSNR:\s*([\d.]+)")
    pattern_SSIM = re.compile(r"SSIM:\s*([\d.]+)")
    pattern_LPIPS = re.compile(r"LPIPS:\s*([\d.]+)")
    pattern_GS = re.compile(r"Number of GS:\s*(\d+)")
    pattern_GS_init = re.compile(r"Model initialized. Number of GS:\s*(\d+)")

    # Extract matches
    #computational_time = int(match.group(1)) if (match := re.search(pattern_time, log_text)) else None
    #PSNR = float(match.group(1)) if (match := re.search(pattern_PSNR, log_text)) else None
    #SSIM = float(match.group(1)) if (match := re.search(pattern_SSIM, log_text)) else None
    #LPIPS = float(match.group(1)) if (match := re.search(pattern_LPIPS, log_text)) else None
    #GS = int(match.group(1)) if (match := re.search(pattern_GS, log_text)) else None

    # Extract matches
    computational_time = get_last_match(pattern_time, log_text, int)
    PSNR = get_last_match(pattern_PSNR, log_text, float)
    SSIM = get_last_match(pattern_SSIM, log_text, float)
    LPIPS = get_last_match(pattern_LPIPS, log_text, float)
    GS = get_last_match(pattern_GS, log_text, int)

    GS_init = get_last_match(pattern_GS_init, log_text, int)
    # Write metrics to file
    with open(args.metrics_file, "a") as f:
        result_str = f"{args.reconstructor_type} {args.scene} {computational_time} {GS_init} {GS} {PSNR} {SSIM} {LPIPS}\n"
        f.write(result_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstructor_type", type=str)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--metrics_file", type=str, default="metrics.txt")

    args = parser.parse_args()
    main(args)
