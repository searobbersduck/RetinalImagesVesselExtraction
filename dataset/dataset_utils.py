import numpy as np
import random

def extract_random(full_imgs,full_masks, patch_h,patch_w, N_patches, inside=True):
    if (N_patches%full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of 20")
        exit()
    assert (len(full_imgs.shape)==4 and len(full_masks.shape)==3)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    # assert (full_masks.shape[3]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[1] == full_masks.shape[1])
    patches = np.empty((N_patches, patch_h, patch_w, full_imgs.shape[3]))
    patches_masks = np.empty((N_patches, patch_h, patch_w))
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    patch_per_img = int(N_patches / full_imgs.shape[0])
    print("patches per full image: {}".format(patch_per_img))
    iter_tot = 0
    for i in range(full_imgs.shape[0]):  #loop over the full images
        k=0
        while k <patch_per_img:
            x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            #check whether the patch is fully contained in the FOV
            # if inside==True:
            #     if is_patch_inside_FOV(x_center,y_center,img_w,img_h,patch_h)==False:
            #         continue
            patch = full_imgs[i,y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2), :]
            patch_mask = full_masks[i, y_center-int(patch_h/2):y_center+int(patch_h/2),x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot]=patch
            patches_masks[iter_tot]=patch_mask
            iter_tot +=1   #total
            k+=1  #per full_img
    return patches, patches_masks

#Divide all the full_imgs in pacthes
def extract_ordered_with_mask(full_imgs, full_masks, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print ("warning: {}".format(N_patches_h) +" patches in height, with about " +''.format(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: {}".format(N_patches_w) +" patches in width, with about " +''.format(img_w%patch_w) +" pixels left over")
    print("number of patches per image: {}".format(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,patch_h,patch_w, full_imgs.shape[3]))
    patches_masks = np.empty((N_patches_tot, patch_h, patch_w))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w,:]
                patch_mask = full_masks[i,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w]
                patches[iter_tot]=patch
                patches_masks[iter_tot] = patch_mask
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches, patches_masks  #array with all the full_imgs divided in patches

#Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, full_masks, patch_h, patch_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    N_patches_h = int(img_h/patch_h) #round to lowest int
    if (img_h%patch_h != 0):
        print ("warning: {}".format(N_patches_h) +" patches in height, with about " +''.format(img_h%patch_h) +" pixels left over")
    N_patches_w = int(img_w/patch_w) #round to lowest int
    if (img_h%patch_h != 0):
        print("warning: {}".format(N_patches_w) +" patches in width, with about " +''.format(img_w%patch_w) +" pixels left over")
    print("number of patches per image: {}".format(N_patches_h*N_patches_w))
    N_patches_tot = (N_patches_h*N_patches_w)*full_imgs.shape[0]
    patches = np.empty((N_patches_tot,patch_h,patch_w, full_imgs.shape[3]))

    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i,h*patch_h:(h*patch_h)+patch_h,w*patch_w:(w*patch_w)+patch_w,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches  #array with all the full_imgs divided in patches

def extract_ordered_overlap(full_imgs, patch_h, patch_w,stride_h,stride_w):
    assert (len(full_imgs.shape)==4)  #4D arrays
    assert (full_imgs.shape[3]==1 or full_imgs.shape[3]==3)  #check the channel is 1 or 3
    img_h = full_imgs.shape[1]  #height of the full image
    img_w = full_imgs.shape[2] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]
    print("Number of patches on h : {}".format(((img_h-patch_h)//stride_h+1)))
    print("Number of patches on w : {}".format(((img_w-patch_w)//stride_w+1)))
    print("number of patches per image: {}".format(N_patches_img) +", totally for this dataset: {}".format(N_patches_tot))
    patches = np.empty((N_patches_tot,patch_h,patch_w, full_imgs.shape[3]))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  #loop over the full images
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                patch = full_imgs[i,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w,:]
                patches[iter_tot]=patch
                iter_tot +=1   #total
    assert (iter_tot==N_patches_tot)
    return patches

def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape)==4)  #4D arrays
    assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h-patch_h)//stride_h+1
    N_patches_w = (img_w-patch_w)//stride_w+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: {}".format(N_patches_h))
    print("N_patches_w: {}".format(N_patches_w))
    print("N_patches_img: {}".format(N_patches_img))
    assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are ".format(N_full_imgs) +" full images (of {}".format(img_h)+"x{}".format(img_w) +" each)")
    full_prob = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))  #itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs,preds.shape[1],img_h,img_w))

    k = 0 #iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//stride_h+1):
            for w in range((img_w-patch_w)//stride_w+1):
                full_prob[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=preds[k]
                full_sum[i,:,h*stride_h:(h*stride_h)+patch_h,w*stride_w:(w*stride_w)+patch_w]+=1
                k+=1
    assert(k==preds.shape[0])
    assert(np.min(full_sum)>=1.0)  #at least one
    final_avg = full_prob/full_sum
    print(final_avg.shape)
    assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg