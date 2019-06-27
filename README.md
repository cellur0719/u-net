# u-net
**1.** Resize the images(**/data/original**) to (256, 256)  and save them as **/data/resize/** (as input images) by **resize.m**.

**2.** Add motion blur to images by **imageProcess.m**. Then get the blurred image in **/data/blur/**.

**3.** Resize the images to (100, 100) and save then as **/data/resize2/** (as output images) by **resize.m**.

**4.** Implement data augmentation by **dataPrepare.ipynb** and save the augmented images as **/data/aug/**

**4.** Train the U-Net by **trainUnet.ipynb**

---

**model.py**: the U-Net Model

**daya.py**: create the data class

