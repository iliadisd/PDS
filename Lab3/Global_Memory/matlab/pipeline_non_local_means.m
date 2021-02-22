  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/monalisa.mat';
  strImgVar = 'monalisa';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [3 3];
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  saveas(gcf,'Original.png');
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  saveas(gcf,'noise.png');
  %% PARAMETERS (Change the dimensions to "play" with images)
  
  threadsPerBlock = [16 16];
  m = 128;
  n = 128;

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/global_kernel.ptx', ...
                               '../cuda/global_kernel.cu');
  
  fprintf('...after cuda kernel %s...\n',mfilename);  
  numberOfBlocks  = [1];
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  %% DATA
  regionx = 3;
  regiony = 3;
  threads = threadsPerBlock(1);
  %% Gaussian Patch
  
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  
  A = single(J);
  Zero = zeros(m,n);
  B = Zero;
  B = single(Zero);
  H = single(H);
  H = H';
  fprintf('...after setting arrays %s...\n',mfilename);  
  %% Get the image with noise to the GPU
  
  tic
  A = gpuArray(A);
  B = gpuArray(B);
  H = gpuArray(H);
  wait(gpuDevice);
  toc

  fprintf('...finish transfer of arrays to GPU %s...\n',mfilename);

  tic
  B =  feval(k, A, B, H, m, n, threads, filtSigma, regionx, regiony) ;
  wait(gpuDevice);
  toc
  tic
  X = gather(B)
  toc
  %% VISUALIZE RESULT
  
  figure('Name', 'Filtered image');
  imagesc(X); axis image; 
  colormap gray;
  saveas(gcf,'Filtered.png');
  
  figure('Name', 'Residual');
  imagesc(X-J); axis image;
  colormap gray;
  saveas(gcf,'Residual.png');
  
  %% (END)

  fprintf('...end %s...\n',mfilename);
