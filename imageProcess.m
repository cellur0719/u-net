Folder = '/Users/shupan/study/summerResearch/selfImplementation/data/resize';
outputFolder = '/Users/shupan/study/summerResearch/selfImplementation/data/blur';
dinfo = dir('*.jpg');
scale = 1;

for K = 1 : length(dinfo)
  sigma = round(10*rand());
  thisimage = dinfo(K).name;
  Img   = imread(thisimage);
  kernelsize = ceil(sigma*3)*2+2;
  kernel     = fspecial('motion',kernelsize,sigma);
  blur_HR    = imfilter(Img, kernel,'replicate');
  LR         = imresize(blur_HR, 1/scale, 'nearest');
  imwrite(LR, fullfile(outputFolder, strcat('blur', thisimage)));  % [EDITED]
end


