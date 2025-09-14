#!/usr/bin/env python3

import os
import shutil
import argparse

def move_unique_files_recursive(src_root, dst_root):
    if not os.path.isdir(src_root):
        raise ValueError(f"Source directory '{src_root}' does not exist.")
    if not os.path.exists(dst_root):
        print(f"Destination root '{dst_root}' does not exist. Creating it.")
        os.makedirs(dst_root)

    moved = 0

    for dirpath, dirnames, filenames in os.walk(src_root):
        rel_path = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_path)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        dst_files = set(os.listdir(dst_dir))

        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_dir, filename)

            if filename not in dst_files:
                try:
                    shutil.move(src_file, dst_file)
                    print(f"Moved: {src_file} → {dst_file}")
                    moved += 1
                except Exception as e:
                    print(f"Error moving {src_file}: {e}")
            else:
                print(f"Skipping existing: {dst_file}")

    print(f"\nDone. {moved} files moved.")

def main():
    parser = argparse.ArgumentParser(description="Recursively move unique files from source to destination.")
    parser.add_argument("src", help="Source root directory (e.g., path1)")
    parser.add_argument("dst", help="Destination root directory (e.g., path2)")
    args = parser.parse_args()

    move_unique_files_recursive(args.src, args.dst)

if __name__ == "__main__":
    main()
