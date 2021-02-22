  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/tiger.mat';
  strImgVar = 'tiger';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
				
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [7 7];
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

  Reg_x = (patchSize(1)) ;
  Reg_y = (patchSize(2)) ;	
  m=256;
  n=256;
  D = floor(Reg_x/2);
  
  %% Mirror padding
  
  for i=1:(m+2*(D))
  	for j=1:(n+2*(D))
		if(i<(1+D) && j<(1+D))
			JN(i,j) = J(i+(D) , j+(D));
		elseif(i>(m+D) && j>(n+D))
			JN(i,j) = J(i-2*(D) , j-2*(D));
		elseif(i<(1+D) && j>(n+D))
			JN(i,j) = J(i+(D) , j-2*(D));
		elseif(i>(m+D) && j<(1+D))
			JN(i,j) = J(i-2*(D) , j+(D));
		elseif(i<(1+D) &&  j<=(m+D) && j>=(1+D))
			JN(i,j) = J(i+(D) , j-D);
		elseif(i>(m+D) &&  j<=(m+D) && j>=(1+D))
			JN(i,j) = J(i-2*(D) , j-D) ;
		elseif(j>(n+D) &&  i<=(m+D) && i>=(1+D))
			JN(i,j) = J(i-D , j-2*(D));
		elseif(j<(1+D) &&  i<=(m+D) && i>=(1+D))
			JN(i,j) = J(i-D , j+(D));
		elseif(i>=(1+D) && i<=(m+D) && j>=(1+D) && j<=(m+D))
			JN(i,j) = J(i-D , j-D); 
	     end
	end	
  end
  
  % Check 
  for i=1:m
	for j=1:n
		JJ(i,j) = JN(i+D,j+D);
	end
  end
  MAXX = max(JJ-J)
  
  %% Threads per Block
  
  threadsPerBlock = [16 16];

  %% (BEGIN)
  
  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/shared_kernel.ptx', ...
                               '../cuda/shared_kernel.cu');
  
  fprintf('...after cuda kernel %s...\n',mfilename);  
  numberOfBlocks  = [16 16];
  
  k.ThreadBlockSize = threadsPerBlock;
  k.GridSize        = numberOfBlocks;
  
  %% DATA
  
  regionx = patchSize(1);
  regiony = patchSize(2);
  
  %% Gaussian Patch
  
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  
  A = single(JN);
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
  B =  feval(k, A, B, H, regionx, regiony, filtSigma) ;
  wait(gpuDevice);
  toc
  tic
  X = gather(B);
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
  
  fprintf('...the biggest error is %s...\n',mfilename);
  Error = max(max(X-J)) 

  fprintf('...the mean value is %s...\n',mfilename);
  Mean = mean(mean(X-J)) 
  %% (END)

  fprintf('...end %s...\n',mfilename);