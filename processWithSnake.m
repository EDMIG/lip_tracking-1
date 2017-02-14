function processWithSnake( path, for_file_name, out_file_name, mat_file )

D = dir(path);

filename = D(1).name;
filename = strcat(for_file_name, filename);
pastI = imread(filename);
%pastI = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking3\liptracking3\liptracking3_01314.jpg');
%pastI = imread('E:\Studies\UCSB\3rd Quarter\Advanced Topics in Computer Vision\Assignments\Assignment2\liptracking4\liptracking4\liptracking3_01314.jpg');

load(mat_file);
template = [x, y; [x(1) y(1)]];

for iForFile = 1:numel(D)
    filename = D(iForFile).name;
    filename = strcat(for_file_name, filename);
    I = imread(filename);
    
    if(size(I, 3) > 1)
        I = rgb2gray(I);
        I = double(I);
    end
    
    [r c] = size(I);
    
    if(iForFile > 1)
        spread = 10;
        
        error = zeros((2*spread + 1) * (2*spread + 1), 3);
        counter = 0;
        
        for xDist = -spread:spread
            for yDist = -spread:spread
                counter = counter + 1;
                n_temp = template;
                n_temp(:, 1) = n_temp(:, 1) + xDist;
                n_temp(:, 2) = n_temp(:, 2) + yDist;
                
                temp_c = round([n_temp(:, 2) n_temp(:, 1)]);
                
                pixelsI1 = zeros(size(temp_c, 1), 1);
                pixelsI2 = zeros(size(temp_c, 1), 1);
                
                for cIterator = 1:size(temp_c, 1)
                    
                    pixelsI1(cIterator) = pastI(temp_c(cIterator, 2), temp_c(cIterator, 1));
                    pixelsI2(cIterator) = I(temp_c(cIterator, 2), temp_c(cIterator, 1));
                    
                end
                
                error(counter, 1) = xDist;
                error(counter, 2) = yDist;
                error(counter, 3) = sum((pixelsI1 - pixelsI2).^2)/numel(pixelsI1);
            end
        end
        
        [val, ind] = min(error(:, 3));
        ind
        newTemp = zeros(size(template, 1), 2);
        newTemp(:, 1) = template(:, 1) + error(ind, 1);
        newTemp(:, 2) = template(:, 2) + error(ind, 2);
        tempCopy = newTemp;
    else
        tempCopy = template;
    end
    
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
    sigma = 0.5;
    %hsize = 5*sigma;
    hsize = 3;
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
    
    EextTot = Eext;
    
%     Cx = gradX;
%     Cy = gradY;
%     Cxx = filter2(sobelX, Cx, 'same');
%     Cyy = filter2(sobelY, Cy, 'same');
%     Cxy = filter2(sobelY, Cx, 'same');
%     Eterm = (Cyy.*Cx.*Cx - 2.*Cxy.*Cx.*Cy + Cxx.*Cy.*Cy)./((Cx.*Cx + Cy.*Cy).^1.5);
%     
%     Eline = I;
%     
%     EextTot = 0.09.*Eline + 1.6.*Eext - 2.*Eterm;


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
    
    
    
    
    upd_template = zeros(size(template));
    
    error = 20;
    th = 0.05;
    
    fx_C = zeros(size(template, 1), 1);
    fy_C = zeros(size(template, 1), 1);
    
    %imshow(I, []);
    %hold on;
    %line(template(:, 1), template(:, 2));
    
    counter = 0;
    while(error > th && counter < 50)
        
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
        
        %k1=inv(finalA);
        
        upd_template(:, 1) = finalA\B_x;
        upd_template(:, 2) = finalA\B_y;
        
        diff = tempCopy - upd_template;
        diff = diff.^2;
        diff_sum = diff(:, 1) + diff(:, 2);
        error = sum(diff_sum);
        %error = 0;
        tempCopy = upd_template;
        
    end
    
    %pastI = I;
    f = figure('visible','off');
    imshow(I, []);
    hold on;
    line(upd_template(:, 1), upd_template(:, 2));
    iForFile
    saveas(f, strcat(out_file_name, int2str(iForFile)),'jpeg');
    
end
end
