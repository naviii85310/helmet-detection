"""
Helmet Detection System - GitHub Push Helper
============================================
Pushes the committed repository files to:
https://github.com/naviii85310/helmet-detection

Usage:
    # 1. Interactive prompt for GitHub Personal Access Token (PAT):
    python push_to_github.py

    # 2. Or provide token directly via command-line:
    python push_to_github.py --token YOUR_GITHUB_TOKEN
"""

import sys
import argparse
import getpass
import dulwich.porcelain as porcelain

REPO_URL = "https://github.com/naviii85310/helmet-detection.git"

def push_repo(token=None, username=None):
    print("=" * 65)
    print("  HELMETVISION AI - GITHUB REPOSITORY PUSH TOOL")
    print(f"  Target: {REPO_URL}")
    print("=" * 65 + "\n")

    if not token:
        print("GitHub requires authentication to push to this repository.")
        print("You can generate a token at: https://github.com/settings/tokens")
        print("Permissions required: 'repo' (Full control of private/public repositories)\n")
        
        username = input("Enter your GitHub Username (e.g. naviii85310): ").strip() or "naviii85310"
        token = getpass.getpass("Enter your GitHub Personal Access Token: ").strip()

    if not token:
        print("[Error] No token provided. Aborting push.")
        sys.exit(1)

    # Embed authentication into URL: https://username:token@github.com/...
    auth_url = f"https://{username or 'naviii85310'}:{token}@github.com/naviii85310/helmet-detection.git"

    try:
        repo = porcelain.open_repo(".")
        print(f"[Git] Staging any remaining files...")
        porcelain.add(repo, paths=["."])
        
        try:
            porcelain.commit(repo, message="Update: HelmetVision AI complete system (frontend, backend, model)")
            print("[Git] Created new commit.")
        except Exception:
            print("[Git] Working directory clean, ready to push.")

        print(f"[Git] Pushing branch 'main' to GitHub...")
        porcelain.push(repo, auth_url, "refs/heads/main")
        print("\n" + "=" * 65)
        print("  SUCCESS! ALL FILES PUSHED TO GITHUB REPOSITORY:")
        print("  👉 https://github.com/naviii85310/helmet-detection")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[Error] Push failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify your token has 'repo' or 'write:packages' scope enabled.")
        print("2. Ensure you have write permissions to 'naviii85310/helmet-detection'.")
        print("3. Alternatively, if you have Git installed, run: git push -u origin main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Helmet Detection project to GitHub")
    parser.add_argument("--token", type=str, default="", help="GitHub Personal Access Token")
    parser.add_argument("--user", type=str, default="naviii85310", help="GitHub username")
    args = parser.parse_args()

    push_repo(token=args.token, username=args.user)
