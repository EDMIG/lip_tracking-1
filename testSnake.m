clear all;
clc;

%I = imread('test.jpg');
I = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking3\liptracking3\liptracking3_01295.jpg');

if(size(I, 3) > 1)
    I = rgb2gray(I);
    I = double(I);
end

[r c] = size(I);

%figure;
%imshow(I, []);
%hold on;
%[x, y] = ginput;

%save('testInputs.mat', 'x', 'y');
%exit;

load('templateFor2.mat');

template = [x, y; [x(1) y(1)]];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%The parameters
alpha = 0.1; %elasticity parameter
beta = 8; %curvature parameter
delta = 1000; %external energy
gamma = 1; %step size
deltaT = 0.05;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Gaussian smoothing
%%%External energy
sigma = 3;
hsize = 5*sigma;
%hsize = 3;
gFilter = fspecial('gaussian', hsize, sigma);
gSmoothed = filter2(gFilter, double(I), 'same');
sobelX = [-1 0 1; -2 0 2; -1 0 1];
sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
gradX = filter2(sobelX, gSmoothed, 'same');
gradY = filter2(sobelY, gSmoothed, 'same');
magGrad = (gradX.^2 + gradY.^2).^0.5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Energy terms
Eext = -1.*magGrad;

Cx = gradX;
Cy = gradY;
Cxx = filter2(sobelX, Cx, 'same');
Cyy = filter2(sobelY, Cy, 'same');
Cxy = filter2(sobelY, Cx, 'same');
Eterm = (Cyy.*Cx.*Cx - 2.*Cxy.*Cx.*Cy + Cxx.*Cy.*Cy)./((Cx.*Cx + Cy.*Cy).^1.5);

Eline = I;

EextTot = 0.09.*Eline + 1.6.*Eext - 2.*Eterm;

fx = filter2(sobelX, EextTot, 'same');
fy = filter2(sobelY, EextTot, 'same');

%Forming D2 and D4
D2 = zeros(max(size(template)));
[dMR dMC] = size(D2);
D4 = D2;

D2 = diag(-2.*ones(size(D2, 1), 1), 0);
D2 = D2 + diag(ones(size(D2, 1) - 1, 1), 1);
D2 = D2 + diag(ones(size(D2, 1) - 1, 1), -1);
D2(1, dMC) = 1;
D2(size(D2, 1), 1) = 1;

D4 = diag(6.*ones(size(D4, 1), 1), 0);
D4 = D4 + diag(-4.*ones(size(D4, 1) - 1, 1), 1);
D4 = D4 + diag(-4.*ones(size(D4, 1) - 1, 1), -1);
D4 = D4 + diag(1.*ones(size(D4, 1) - 2, 1), 2);
D4 = D4 + diag(1.*ones(size(D4, 1) - 2, 1), -2);
D4 = D4 + diag(ones(2, 1), dMR - 2) + diag(ones(2, 1), 2-dMR);
D4(1, dMC) = -4;
D4(size(D4, 1), 1) = -4;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Forming A
A = -1 * alpha.*D2 + beta.*D4;
finalA = gamma.*eye(size(D4, 1)) + deltaT.*A;

%Forming B



tempCopy = template;
upd_template = zeros(size(template));

error = 20;
th = 0.05;

fx_C = zeros(size(template, 1), 1);
fy_C = zeros(size(template, 1), 1);

%imshow(I, []);
%hold on;
%line(template(:, 1), template(:, 2));

counter = 0;
while(error > th && counter < 500)
    
    counter = counter + 1;
    
    for i = 1:size(tempCopy, 1)
        xCord = tempCopy(i, 1);
        yCord = tempCopy(i, 2);
        
        if(xCord < 0 || yCord < 0)
            1
            tempCopy(i, 1) = c;
            tempCopy(i, 2) = r;
            fx_C(i) = fx(r, c);
            fy_C(i) = fy(r, c);
        elseif((mod(10*xCord, 10) == 0 || mod(10*xCord, 10) > 9) && (mod(10*yCord, 10) == 0 || mod(10*yCord, 10) > 9))
            fx_C(i) = fx(uint8(yCord), uint8(xCord));
            fy_C(i) = fy(uint8(yCord), uint8(xCord));
            2
        elseif(xCord > c || yCord > r)
            tempCopy(i, 1) = c;
            tempCopy(i, 2) = r;
            indices = [c r];
            fx_C(i) = fx(r, c);
            fy_C(i) = fy(r, c);
        else
            3
            ceil_x = ceil(xCord);
            ceil_y = ceil(yCord);
            floor_x = floor(xCord);
            floor_y = floor(yCord);
            
            indices = [floor_x floor_y; floor_x ceil_y; ceil_x floor_y; ceil_x ceil_y];
            
            pixel_mat_x = [fx(floor_y, floor_x) fx(floor_y, ceil_x) fx(ceil_y, floor_x) fx(ceil_y, ceil_x)];
            pixel_mat_y = [fy(floor_y, floor_x) fy(floor_y, ceil_x) fy(ceil_y, floor_x) fy(ceil_y, ceil_x)];
            
            fx_C(i) = sum(pixel_mat_x)/numel(pixel_mat_x);
            fy_C(i) = sum(pixel_mat_y)/numel(pixel_mat_y);
            
        end
        
    end

    B_x = gamma.*tempCopy(:, 1) - deltaT.*fx_C;
    B_y = gamma.*tempCopy(:, 2) - deltaT.*fy_C;
    
    k1=inv(finalA);
    
    upd_template(:, 1) = finalA\B_x;
    upd_template(:, 2) = finalA\B_y;
    
    diff = tempCopy - upd_template;
    diff = diff.^2;
    diff_sum = diff(:, 1) + diff(:, 2);
    error = sum(diff_sum);
    %error = 0;
    tempCopy = upd_template;
    
    %line(upd_template(:, 1), upd_template(:, 2));
    %line(tempCopy(:, 1), tempCopy(:, 2));
end

imshow(I, []);
hold on;
line(upd_template(:, 1), upd_template(:, 2));

% f = figure('visible','off');
% imshow(I, []);
% hold on;
% line(upd_template(:, 1), upd_template(:, 2));
% pathName = 'E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking2\output\output';
% strcat(pathName, 'output');
% saveas(f, pathName,'jpeg');

















%{
figure;
imshow(i);
hold on;
[x, y] = ginput;

save('testInputs.mat', 'x', 'y');
%}

%imshow(i);
%hold on;
%plot(x, y, 'c+');
%line(x, y);


