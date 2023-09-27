import gzip
import shutil
import os
import fnmatch


def gunzip(file_path, output_path):
    with gzip.open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def recurse_and_gunzip(root):
    walker = os.walk(root)
    print(list(walker))
    for root, dirs, files in walker:
        print(root, dirs, files)
        for f in files:
            print(f, root)
            if fnmatch.fnmatch(f, "*.gz"):
                print("gunzipping {}".format(f))
                gunzip(root + "/" + f, root + "/" + f.replace(".gz", ""))


recurse_and_gunzip('data')

