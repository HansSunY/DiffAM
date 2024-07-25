import torch
from torch.autograd import Variable
from torchvision import transforms
import PIL
from PIL import Image
import copy
def get_target_image(target_id):
    transform =  transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if target_id == 0:
        target_image = Image.open("assets/datasets/target/005869.jpg").convert('RGB')
        target_image = (transform(target_image).to("cuda").unsqueeze(0))
        test_image = Image.open("assets/datasets/test/008793.jpg").convert('RGB')
        test_image = (transform(test_image).to("cuda").unsqueeze(0))
        target_name = "005869"
    elif target_id == 1:
        target_image = Image.open("assets/datasets/target/085807.jpg").convert('RGB')
        target_image = (transform(target_image).to("cuda").unsqueeze(0))
        test_image = Image.open("assets/datasets/test/047073.jpg").convert('RGB')
        test_image = (transform(test_image).to("cuda").unsqueeze(0))
        target_name = "085807"
    elif target_id == 2:
        target_image = Image.open("assets/datasets/target/116481.jpg").convert('RGB')
        target_image = (transform(target_image).to("cuda").unsqueeze(0))
        test_image = Image.open("assets/datasets/test/055622.jpg").convert('RGB')
        test_image = (transform(test_image).to("cuda").unsqueeze(0))
        target_name = "116481"
    else:
        target_image = Image.open("assets/datasets/target/169284.jpg").convert('RGB')
        target_image = (transform(target_image).to("cuda").unsqueeze(0))
        test_image = Image.open("assets/datasets/test/166607.jpg").convert('RGB')
        test_image = (transform(test_image).to("cuda").unsqueeze(0))  
        target_name = "169284"
    return target_image, test_image, target_name
     
def get_ref_image(ref_id):
    train_transform = transforms.Compose([transforms.Resize([256,256]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),inplace=True)])
    mask_transform = transforms.Compose([transforms.Resize((256,256),interpolation=PIL.Image.NEAREST),ToTensor])
    makeup_image = Image.open('assets/datasets/pairs/'+ref_id+'_m.png')
    non_makeup_image = Image.open('assets/datasets/pairs/'+ref_id+'_nm.png')
    makeup_mask = Image.open('assets/datasets/pairs/'+ref_id+'_mask.png')
    makeup_image = train_transform(makeup_image).to("cuda")
    non_makeup_image = train_transform(non_makeup_image).to("cuda")
    makeup_mask = mask_transform(makeup_mask)
    return makeup_image, non_makeup_image, makeup_mask

def cal_hist(image):
    """
        cal cumulative hist for channel list
    """
    hists = []
    for i in range(0, 3):
        channel = image[i]
        # channel = image[i, :, :]
        channel = torch.from_numpy(channel)
        # hist, _ = np.histogram(channel, bins=256, range=(0,255))
        hist = torch.histc(channel, bins=256, min=0, max=256)
        hist = hist.numpy()
        # refHist=hist.view(256,1)
        sum = hist.sum()
        pdf = [v / sum for v in hist]
        for i in range(1, 256):
            pdf[i] = pdf[i - 1] + pdf[i]
        hists.append(pdf)
    return hists


def cal_trans(ref, adj):
    """
        calculate transfer function
        algorithm refering to wiki item: Histogram matching
    """
    table = list(range(0, 256))
    for i in list(range(1, 256)):
        for j in list(range(1, 256)):
            if ref[i] >= adj[j - 1] and ref[i] <= adj[j]:
                table[i] = j
                break
    table[255] = 255
    return table


def histogram_matching(dstImg, refImg, index):
    """
        perform histogram matching
        dstImg is transformed to have the same the histogram with refImg's
        index[0], index[1]: the index of pixels that need to be transformed in dstImg
        index[2], index[3]: the index of pixels that to compute histogram in refImg
    """
    index = [x.cpu().numpy() for x in index]
    dstImg = dstImg.detach().cpu().numpy()
    refImg = refImg.detach().cpu().numpy()
    dst_align = [dstImg[i, index[0], index[1]] for i in range(0, 3)]
    ref_align = [refImg[i, index[2], index[3]] for i in range(0, 3)]
    hist_ref = cal_hist(ref_align)
    hist_dst = cal_hist(dst_align)
    tables = [cal_trans(hist_dst[i], hist_ref[i]) for i in range(0, 3)]

    mid = copy.deepcopy(dst_align)
    for i in range(0, 3):
        for k in range(0, len(index[0])):
            dst_align[i][k] = tables[i][int(mid[i][k])]

    for i in range(0, 3):
        dstImg[i, index[0], index[1]] = dst_align[i]

    dstImg = torch.FloatTensor(dstImg).cuda()
    return dstImg

def to_var(x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)
        
def de_norm(x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

def criterionHis(input_data, target_data,index,criterionL1,mask_src=1, mask_tar=1,):
        input_data = (de_norm(input_data) * 255).squeeze()
        target_data = (de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        # dstImg = (input_masked.data).cpu().clone()
        # refImg = (target_masked.data).cpu().clone()
        input_match = histogram_matching(input_masked, target_masked, index)
        input_match = to_var(input_match, requires_grad=False)
        loss = criterionL1(input_masked, input_match)
        return loss,input_match/255

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def rebound_box(mask_A, mask_B, mask_A_face):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 1]
        y_A_index = index_tmp[:, 2]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 1]
        y_B_index = index_tmp[:, 2]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11] =\
                            mask_A_face[: ,min(x_A_index)-10:max(x_A_index)+11, min(y_A_index)-10:max(y_A_index)+11]
        mask_B_temp[: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11] =\
                            mask_A_face[: ,min(x_B_index)-10:max(x_B_index)+11, min(y_B_index)-10:max(y_B_index)+11]
        mask_A_temp = to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp    

def mask_preprocess(mask_A, mask_B):
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 1]
        y_A_index = index_tmp[:, 2]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 1]
        y_B_index = index_tmp[:, 2]
        mask_A = to_var(mask_A, requires_grad=False)
        mask_B = to_var(mask_B, requires_grad=False)
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2