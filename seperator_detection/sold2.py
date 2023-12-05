import kornia as K
import kornia.feature as KF
import torch
from PIL import Image, ImageDraw

def line_detect(image_path):
    torch_img1 = K.io.load_image(image_path, K.io.ImageLoadType.RGB32)[None, ...]
    torch_img1_gray = K.color.rgb_to_grayscale(torch_img1)
    imgs = torch.cat([torch_img1_gray], dim=0)
    sold2 = KF.SOLD2(pretrained=True, config=None)
    outputs = sold2(imgs)
    line_seg1 = outputs["line_segments"][0]
    desc1 = outputs["dense_desc"][0]
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image, "RGB")
    for line in line_seg1:
        draw.line((line[0][0].item(),
                   line[0][1].item(),
                   line[1][0].item(),
                   line[1][0].item(),), 'green', width=4)
    image.save('temp/sold2.png')
    print()

if __name__ == "__main__":
    line_detect(r'sep_dataset/test/fi/images/576443_0001_23676242.jpg')

