import torch as T
from PIL import Image, ImageOps
import os
from joblib import Parallel, delayed

images_folder = "data/Original/Images"

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def get_image(opt, info, istrain):
    from_w, from_h, ind = info[0], info[1], info[2]
    if istrain:
        filename = opt.train_files[ind][0][0]
    else:
        filename = opt.test_files[ind][0][0]

    img = Image.open(os.path.join(images_folder, filename)).convert('RGB')
    width, height = img.size
    size = min(width, height)
    img = img.crop((from_w, from_h, from_w + size, from_h + size))
    return img, size


def retina(l, info, opt, istrain):
    l = l.cpu().numpy()

    def extract_patches_batch(l, size):                     #Extract square patches for given batch with given location as center and given size as length
        batch_size = len(l)

        def get_patch(i):                                   #Get patch for each datapoint in a batch
            img, imgsize = get_image(opt, info[i], istrain) #Get context image
            patch_size = imgsize // 4
            patch_size *= size                              #original size of patch before compressing it to 96x96
            #location resized from [-1,1] to [image_size, image_size]
            l_denorm = (0.5 * imgsize * (1 + l[i])).astype(int)
            from_x, from_y = l_denorm[0] - (patch_size // 2), l_denorm[1] - (patch_size // 2)
            to_x, to_y = from_x + patch_size, from_y + patch_size
            #pad context image if corners of the patch exceeds its borders
            if (from_x < 0 or from_y < 0 or to_x > imgsize or to_y > imgsize):
                temp = patch_size // 2 + 1
                img = ImageOps.expand(img, border=temp, fill='black')
                from_x += temp
                from_y += temp
                to_x += temp
                to_y += temp

            img = img.crop((from_x, from_y, to_x, to_y))
            img = opt.my_transform(img).unsqueeze(0)
            return img

        patches = Parallel(n_jobs=opt.n_jobs, backend="threading")(
            delayed(get_patch)(i) for i in range(batch_size)    #Parallelize get_patch function as its execution for each datapoint in batch is independent of others
        )
        patches = get_cuda(T.cat(patches, dim=0))
        return patches

    phi = []
    size = opt.start_size

    for i in range(opt.k):
        phi.append(extract_patches_batch(l, size))
        size *= 2

    return phi

def get_color(i):
    if i == 0:
        return "red"
    elif i == 1:
        return "blue"
    return "green"