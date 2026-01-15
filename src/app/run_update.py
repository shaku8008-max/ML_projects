import subprocess
import sys

def run(cmd: str):
    print("\n>>>", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("\n❌ Failed:", cmd)
        sys.exit(result.returncode)

def main():
    # 1) Download latest EPL data
    run("python3 src/injest/download_epl.py")

    # 2) Build features
    run("python3 src/features/epl_add_binary_target.py")
    run("python3 src/features/epl_build_elo_homeadv.py")
    run("python3 src/features/epl_build_form_last5_last10_gd.py")

    # 3) Optional split
    run("python3 src/features/epl_split_70_30_v2.py")

    print("\n✅ DONE: EPL data updated and features rebuilt.")
    print("Check data/processed/ for latest files.")

if __name__ == "__main__":
    main()
