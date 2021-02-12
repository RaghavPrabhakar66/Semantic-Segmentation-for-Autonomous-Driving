import matplotlib.pyplot as plt

def convertFromTensor(imageTensor):
    x = imageTensor.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)

    return x

def show_imgs(img, mask):
    #print(img.shape, convertFromTensor(img).shape)
    #img = np.rollaxis(convertFromTensor(img),0,3)
    #mask  = np.rollaxis(convertFromTensor(mask),0,3)

    img  = convertFromTensor(img)
    mask = convertFromTensor(mask)

    fig = plt.figure(figsize=(15,20))

    ax = fig.add_subplot(1,2,1)
    ax.imshow(img)
    ax.set_title('Image')

    ax1 = fig.add_subplot(1,2,2)
    ax1.imshow(mask)
    ax1.set_title('Ground Truth')