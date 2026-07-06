% Input Images,
BackgroundImageFileName = 'poisson_editing/source.jpg';  % some example images are provided
ObjectImageFileName     = 'poisson_editing/sunset.png'; % in the "./images" folder.
OmegaImageFileName      = 'poisson_editing/mask.jpg';

% Location of the centroid of Omega in the background image,
x0 = 440;
y0 = 280;

mode = 'Average'; % 'Replace','Max','Average'.

solver = 'I';

% Read input images
OIm    = imread(BackgroundImageFileName);
BIm    = imread(ObjectImageFileName);
Omega  = imread(OmegaImageFileName);

if size(Omega,3)==1&&size(OIm,3)==3, Omega=cat(3,Omega,Omega,Omega); end
s = get(0,'ScreenSize');

BIm   = double(BIm); OIm = double(OIm);
Omega = double(Omega==255);

Omega0             = Omega;
OIm0               = OIm;
[Ho,Wo,Co]         = size(Omega0);
[H,W,C]            = size(BIm);
Omega              = zeros(H,W,C);
OIm                = zeros(H,W,C);

Omega(1:Ho,1:Wo,:) = Omega0;
Omega0             = mean(Omega0,3);
OIm(1:Ho,1:Wo,:)   = OIm0;

[X,Y]              = meshgrid(1:Wo,1:Ho);
xg                 = mean(X(Omega0(:)==1));
yg                 = mean(Y(Omega0(:)==1));
Omega              = circshift( Omega, round([y0-yg x0-xg]) );
OIm                = circshift( OIm  , round([y0-yg x0-xg]) );
clear xg yg X Y Omega0 OIm0

% 2 Comput and combine gradients

% 2.1 Comput gradients,
Grad_OIm  = ComputeGradient(OIm); % grad. of the "Object"
Grad_BIm  = ComputeGradient(BIm); % grad. of the "Background"

% 2.2 Combine gradients,
G    = CombineGradients(Grad_BIm, Grad_OIm, Omega, mode);
clear Grad_OIm Grad_BIm


% 3 Solve poisson equation using Fourier
I = SolvePoissonEq_I(G.x,G.y);
BIm_outside_Omega = BIm.*(1-Omega);
I_outside_Omega   = I.*(1-Omega);
for c = 1:C,
    % mean value of the input image, (outside Omega)
    input_mean_value = sum(sum(BIm_outside_Omega(:,:,c))) / ...
                       sum(sum(1-Omega(:,:,c)));
    % mean value of the output image, (outside Omega)
    out_mean_value   = sum(sum(I_outside_Omega(:,:,c))) /...
                       sum(sum(1-Omega(:,:,c)));
    % Set the mean value,
    I(:,:,c)   = I(:,:,c) - out_mean_value + input_mean_value;
end
clear c BIm_outsideOmega I_outsideOmega ...
      input_mean_value out_mean_value


% Display Results
[H,W,c] = size(I); name = 'I (output image)';
figure('Color',[1 1 1],'MenuBar','none',...
       'Position',[s(3)-W s(4)-H+40 W H],'Name',name, ...
       'NumberTitle','off'); imshow(uint8(I))
