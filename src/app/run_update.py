import subprocess
import sys

def run(cmd: str):
    print("\n>>>", cmd)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("\n❌ Failed:", cmd)
        sys.exit(result.returncode)

def main():
    # 1) Download latest EPL CSV
    run("python3 src/injest/download_epl.py")

    # 2) Build features (base dataset)
    run("python3 src/features/epl_build_elo_homeadv.py")
    run("python3 src/features/epl_build_form_last5_last10_gd.py")

    # 3) Build targets
    run("python3 src/features/epl_add_binary_target.py")              # match-winner target (home_win)
    run("python3 src/features/epl_add_btts_winner_target.py")         # ✅ new target

    # 4) Train models
    run("python3 src/model/epl_train_rf_v2.py")                       # match-winner model (if you want)
    run("python3 src/model/train_btts_winner_rf.py")                  # ✅ BTTS+Winner model

    print("\n✅ DONE: Data updated, features rebuilt, targets generated, models trained.")


if __name__ == "__main__":
    main()
