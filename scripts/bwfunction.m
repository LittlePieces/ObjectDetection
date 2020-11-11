function bw = bwfunc(src1, src2)
fold = dir(fullfile(src1,'*.bmp'));
refold = src2;
for kk = 1:length(fold)
    img=imread(strcat(fold(kk).folder,'\',fold(kk).name));
    re=bwfunc(img);
    m =mean2(re);
    s = std2(re);
    maxv = max(re(:));
    T = m + 0.5*(maxv - m);
    bw = re> T;
    imwrite(re, strcat(refold,'\',fold(kk).name));
end