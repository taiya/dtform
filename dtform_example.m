clc, clear, close all;
[XX,YY] = meshgrid(-1.5:0.01:1.5,-1:0.01:1);
% imagesc(XX), axis equal;

% I = double( ((XX-.5).^2 + (YY).^2 - .2^2)<0 );

I0 = ((XX+.2).^2 + (YY-.3).^2 - .2^2) < 0;
I1 = ((XX-.3).^2 + (YY+.4).^2 - .2^2) < 0;
I = I0 + I1;

% imagesc(I), axis equal;

mex CXXFLAGS='$CXXFLAGS -std=c++11' dtform.cpp; 
tic()
[I2, CORRS] = dtform(I);
toc();
figure, imagesc(I2), axis equal;
