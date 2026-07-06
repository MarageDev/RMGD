
function GradF = ComputeGradient(F)
% Input,
%   - F: (HxWxC) image (C=1 gray, C=3 color image)
% Output,
%   - GradF: struct where GradF.x -> (HxWxC) x-partial derivative of F
%                     and GradF.y -> (HxWxC) y-partial derivative of F

F = double(F);


% First Expand the domain and the image,
F = [F F(:,end:-1:1,:)]; F = [F; F(end:-1:1,:,:)];
[H,W,C] = size(F);
GradF.x = zeros(H,W,C); GradF.y = zeros(H,W,C);
% initialization
i       = sqrt(-1); % imaginary unit,
ft      = @(U) fftshift(fft2(U)); % 2D-Fourier transform,
ift     = @(U) real(ifft2(ifftshift(U))); % inv. Fourier trans.,
[Jc,Ic] = meshgrid( 1:W , 1:H );      % define the spatial
j0 = floor(W/2)+1; i0 = floor(H/2)+1; % frequencies domain
Jc = Jc - j0; Ic = Ic - i0;           % (center)
for c = 1:C,
    GradF.x(:,:,c) = ift( (i*2*pi/W*Jc).*ft(F(:,:,c)) );
    GradF.y(:,:,c) = ift( (i*2*pi/H*Ic).*ft(F(:,:,c)) );
end
GradF.x = GradF.x(1:H/2,1:W/2,:);
GradF.y = GradF.y(1:H/2,1:W/2,:);

end %function
