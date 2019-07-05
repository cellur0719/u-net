# u-net
**1.** Resize the images(**/data/original**) to (572, 572)  and save them as **/data/resize/** (as input images) by **resize.m**.

**2.** Add motion blur to images by **imageProcess.m**. Then get the blurred image in **/data/blur/**.

**3.** Implement data augmentation by **dataPrepare.ipynb** ,save the augmented images as **/data/aug/** and get image patches((200 * 70, 80, 80, 3))

**4.** Train the U-Net by **trainUnet.ipynb**

---

**model.py**: the U-Net Model

**data.py**: create the data class

