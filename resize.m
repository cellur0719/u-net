Folder = '/Users/shupan/study/summerResearch/selfImplementation/data/resize';
outputFolder = '/Users/shupan/study/summerResearch/selfImplementation/data/resize2';
dinfo = dir('*.jpg');

for K = 1 : length(dinfo)
  sigma = round(20*rand());
  thisimage = dinfo(K).name;
  Img   = imread(thisimage);
  Img2 = imresize(Img, [100, 100]);
  imwrite(Img2, fullfile(outputFolder, strcat('re', thisimage)));  % [EDITED]
  imshow(Img2)
end