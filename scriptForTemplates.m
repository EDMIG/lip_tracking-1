clear all;
clc;

%I = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking3\liptracking3\liptracking3_01314.jpg');
%I = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking4\liptracking4\liptracking4_01314.jpg');
I = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking4\liptracking4\liptracking4_00068.jpg');

if(size(I, 3) > 1)
    I = rgb2gray(I);
    I = double(I);
end

[r c] = size(I);

figure;
imshow(I, []);
hold on;
[x, y] = ginput;
close all;

save('templateForThirdSet.mat', 'x', 'y');