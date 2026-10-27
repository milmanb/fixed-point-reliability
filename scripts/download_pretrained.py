"""Download the pretrained denoisers and the BSD68 test images for evaluate_pretrained.py.

    data/pretrained/dncnn/dncnn_gray_blind.pth                    DnCNN-B, KAIR release v1.0 (2.7 MB)
    data/pretrained/restormer/gaussian_gray_denoising_blind.pth   Restormer, Hugging Face (105 MB)
    data/bsd68/test001.png ... test068.png                        BSD68, DnCNN repository (6.5 MB)

data/ is git-ignored. Each file is checked against its SHA-256 digest before it is moved into
place. The Restormer digest is the one its host publishes; the BSD68 files were checked against
their git hashes at the pinned commit, and the DnCNN-B weights against the paper's BSD68 PSNR
(src/fpr/pretrained.py). The BSD68 images come from the Berkeley Segmentation Dataset, whose
terms allow non-commercial research and educational use. Nothing here is redistributed.

    python scripts/download_pretrained.py
"""

import argparse

from fpr.data import REPO_ROOT
from fpr.download import fetch

WEIGHTS = {  # target under data/ -> (URL, SHA-256)
    "pretrained/dncnn/dncnn_gray_blind.pth": (
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth",
        "cc2ac7fa2f248a7f7a3da414a848326634511ec8b5adb7799e857ed788067bfe"),
    "pretrained/restormer/gaussian_gray_denoising_blind.pth": (
        "https://huggingface.co/deepinv/Restormer/resolve/aa2667c79d766b67da2c1489c7ce5f0423bae798/"
        "gaussian_gray_denoising_blind.pth",
        "4d3ab68b4e0e205c37c8cd3d8bb049d951baa6461e3b4dde6bb02b7f1f34deeb"),
}
BSD68_URL = ("https://raw.githubusercontent.com/cszn/DnCNN/55fdeb2ac82c8c8cae6e281dae2b8e87e2d47a4d/"
             "testsets/BSD68")
BSD68 = """
94ddd662377347ca7b07d5a11c2a83feff658723cf31f79bfa75f65713e94ee4  test001.png
7dabadb169c463d7bd7da110c36d486846c8c787883b6fc979257d44e09b3a29  test002.png
d27b8117a1b175b594f80de162e81776c8587e2e512fa4d07fbaeacb67c97949  test003.png
3ceebbbf583e355fefcdb9e653bff65451b20a512f1c4ca12a0518f4569b481c  test004.png
d187d8b58e5d19b4aa6e055b849064fb974cad7b18083c25f6580a1ba17d6bb3  test005.png
c184fc29e6668cbb15ae42c52cb68a36b4179d19ba5dbc4393f395ccc8f14aec  test006.png
d13c69290e9783d2b5c1ed2acd3e51ba5ac3b6c8731fe26141d250a5e3f042e9  test007.png
be55c170e63f28a2e7899687e083b42e6a8c9e465913dfaf4d2b444567856933  test008.png
ba0ad02a3f761ffab2fc23a26705944fc1983d0b51ecff3600d137bec465a98b  test009.png
fbe5bd8cfdfc8232db92e4f31a7cb1acc2f7540a64cbd3bd4aa7d32f67669c15  test010.png
ddbb6e31f8279db81d04215c2868de748c66e31a30f48f066b44c6837ace5c20  test011.png
3a6330f054a1c16e8307446c303648bc87828ba6282af6586482349fc50d19ef  test012.png
28f32a8b20cb9469ad6df196c0b4f92eb73a3c6483904d8a56cb2331a26d1494  test013.png
61c904019033176fb7a41b6a6f6727ffcb11e702b0a4dc14d8b583b41c9d8926  test014.png
38dc56d13f8606ad8a07c88192885f9f2d7015e46006ae250a7c41fa8fa1b136  test015.png
ed5cbdb3402a36eb7032b54fa11bc1845d9bd08a6c95ed621773e0ac56b398f6  test016.png
f0c1604265e5cacfea684562bed7a0d610d72ffa668c0694bf9143271a84242b  test017.png
3540e6c59c0c62f5e673ca4955b8ca8ffe2853cbe51ac82d89f5741755ce464b  test018.png
8bbfcc1d035e25c6bb541215a653ffea2a5b729b367aa2c68b33084c6c3e00b6  test019.png
3775d6efddca1dcca090742ad1448f0fa48a25bc24ce4eca7ed765532c9c3f39  test020.png
0a757d926afa75d83c7bffce2de33e05d6ca12cdef6a4d4187deca751edbf107  test021.png
b797e008601c61ff11dfa936be139ec52a5333d41949cb7ceecd5afb3826bef6  test022.png
6b09da52d43e5ae5d60780439ad3a753e9769efd26045187e9dd63e02d3b46dc  test023.png
67ab076e3b3f1d39182627d2405897a879b09b40ebe6b33e1bf4532e5f746fc5  test024.png
792b825650b4c4e553144dd69de37b519859e023535f5bba317abf5dbbac2957  test025.png
86a6acba42db469bcb716a299cbca5839757798fe8c61db85e62e9278261df1b  test026.png
cc4e573ee4979d06fb04050eb1f9e68f7eebbe43cdd9b75e4627ec2b0c795c18  test027.png
549b2639e72f6825fda7c2cc456cce4ea97ffcbd1f2bca5f6cded351da769d6f  test028.png
49351b50363c087dd1687ad10e7d98804eb81546eb378250cf7549c74c863836  test029.png
bd70a77c744cb56045489afac9efac6749fbf6f1e22abc2cddcbfe947579ae1e  test030.png
671fba87a6c54610728ef8130c36f255c198b70055064cfc5cdc961e08e6c132  test031.png
cc734bc7d9ed080286e4dca3a4f9d242870c76881989c2e0dd3fd83d8dc13468  test032.png
790f437a438f4254762ec4d4a4aefa6c16334fedb5137848154d45ce765ab80b  test033.png
7a7b04301e6b260473baa3fdbbbb04d2b07c6e0bae541d43484402f5202cef5b  test034.png
f10990d3da8a55198f64bce464d59417bcbd719945dd1e95b46c3ed37bef3963  test035.png
bca7b2561ac285d92b624bea36a9b663e3646e38f090e3236e5c7267ab99988c  test036.png
c04809c41445b37b2b17cfbdf0342dfa891c9854a57866dabed27c499ab8a803  test037.png
a076c1f412a65ca94ba712b8158a13b93502732f3e8470a5c1367fc3d1134fe2  test038.png
2c5c513bcc633ca5a3639ce4c45d89cdb400e6748fcd7c668aacaa04f9d045d5  test039.png
f79ef5e81e07c003736d1af6ddb616fb8f29f7747769315adf591ad515e726fc  test040.png
646c912a3e15bcd228ed49f6de5c63a32f82dbbbb36b15529ecddeff91e495e5  test041.png
78cab10bcad2df64cbd0fc17ce0cb5af95ff98cbb1ee27f2664eccea93d555c2  test042.png
5ae7bac9730294377d5ca0e711facff8aa7bdc388d46feb7fa23398d055455ec  test043.png
cedc1ec61bde1eb220abe22d3753546ddae9e2600c38901721c3f7c1d4fd8f3e  test044.png
ba387a966b0980d984bc4a34823ff70f33b207c13c62256f2be7a55a30e9aa9b  test045.png
52ed8a3381605a00da47f1306ce4d2c71c5310bf49c6b9ada2a194146854db53  test046.png
55021821543280a66567f78112fa84e8b6e0c0a21f38d15c013c3de7127f4acf  test047.png
f303f403f3a6574a2f45c7594ce390a80b1677e35ee74b9c31edf54fa8e4f7c8  test048.png
a9864e710ad5e3feaa08ac114bbe05714466b888cb16a689ac2489e2d5b0d7e5  test049.png
24cb8d9565e5469ef6e336cb7ab26f98c85a3b1d09aef0d67e79940a5ef257ac  test050.png
3c12ea89553d44808bc4fb44f9558878cb96cdc994590c877c6d2faca2fc72c4  test051.png
7bf722086b02521cbf254648b24462b97a63869a85d160c1c410bb0fbe8533ac  test052.png
235bbd5e9e72ece8e9cf439d1c95f89d41a94efd5e0e48ff6eaa053e5f0378de  test053.png
ae319876af64a54390e312e11e405eed0ab85707e55c1b54106e5dd25e2406bf  test054.png
ab39e645bd055dc0d4f10ccedad26d54712745828860f9153cfc99094eb38a1e  test055.png
87c7e3c0112781479ae9196952a13c33364168c366b4a546f6bd185c6be7efd6  test056.png
83b32759876dd2b3d024e78823bf414830a2e1433d74f0500cbb6dcd6e0a13e3  test057.png
e321bae2f2802b76ae00b279aa04450434c7b20c0b38ad956cb91f74b3ca9e0f  test058.png
88232cf229ae887c088cc76ce8a0f43e83761432956732fb5f2fb87c691574fd  test059.png
8f900a9ea1da160e442ce5b62ed139fe651f6d35171371314f813a301f3b3c4b  test060.png
09c5e1cb5b17d6073358dd1df9385a7bec3ddaccb34124060de6e30b9fa80481  test061.png
7697d1cb6007fa4d01455508d3436ff936a0e1915a3fb17dbc95ec332a7232c3  test062.png
6b9171d3a2db8e5784a211c2a1d94de02b147acdd04f603d6df6c29db6b38700  test063.png
1f705d2e47aeea31d31bc13cb02024cccdb268ea19adb8b5487d4a899e0a9e1b  test064.png
8ccba9966431cd0928054c49b4b70ffd95a25b821b6d6d6f3de9090bb57781ab  test065.png
473d35176134c1bb2ad3d7e915c5e7f397d625f1245f5d7cb23bdfd46ad36485  test066.png
b045137f029e1c406ba0cec6363d1d09fcfd1ff01298682e98066770cc302fd6  test067.png
26fcaab8eeb61b86e4ed9aa8dcb23c2b6ee652ca3ea7ff545e7e27610dec2c7c  test068.png
"""  # sha256sum format: digest, two spaces, file name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace files that differ")
    return parser.parse_args()


def main():
    args = parse_args()
    data = REPO_ROOT / "data"
    for target, (url, expected) in WEIGHTS.items():
        print(f"data/{target}: {fetch(url, data / target, expected, force=args.force)}")
    statuses = []
    for line in BSD68.split("\n"):
        if line.strip():
            expected, name = line.split()
            statuses.append(fetch(f"{BSD68_URL}/{name}", data / "bsd68" / name, expected,
                                  force=args.force))
    print(f"data/bsd68: {len(statuses)} images, {statuses.count('downloaded')} downloaded, "
          f"{statuses.count('present')} present")


if __name__ == "__main__":
    main()
