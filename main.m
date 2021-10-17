



                                            %-------------------%
    % load AMLALL_nature                      %  loading dataset  %
    % load Lymphoma                         %-------------------%

    %% 
    % num_SelectedGenes = input('How many genes you want to select? \nPlease input here:');
    num_SelectedGenes = 5;


    % mean:0,std:1                           %-------------------%
    X = normalizemeanstd( xapp );            %    normalizing    %
                                             %-------------------%

    %============================================================%
    %                KERNEL PARTIAL LEAST SQUARES                %
    %============================================================%
    Y = binarize( yapp );

    % number of components
    num_Component = 10;
                                           %---------------------%
    alpha = 1;                             %  parameter setting  %
    coef = 0.1;                            %---------------------%
                                                                     %---------------------%
                                                                     %  polynomial kernel  %
%     Kxx = kernel( X, X, 'polynomial', alpha, coef );                 %---------------------%
%     Kxy = kernel( X, X([1:2:size(X,1)], : ), 'polynomial', alpha, coef );                                                                
                                                                    %---------------------%
                                                                    %   gaussian kernel   %                
    Kxx = kernel( X, X, 'gaussian' );  
    %---------------------% 
    Kxy = kernel( X, X([1:2:size(X,1)],:), 'gaussian' );
%     varargin = yapp([1:2:size(X,1)],:)


    [ kplsXS, trainerro ] = kernelPLS( Kxx, Kxy, Y, num_Component);
    trainerro
    trian_erro(i) = trainerro;

%     kX0 = X - ones( size(X,1), 1 )*mean( X );
%     kWeight = pinv( kX0 )*kplsXS;

%     kVIP = calVIP( Y, kplsXS( :, 1:num_Component ), kWeight( :, 1:num_Component ) );
% 
%     [ ~, FeatureRank ] = sort( kVIP, 'descend' );

%     res_idx = FeatureRank(1:20, 1)
%     stay=stay_idx(res_idx)
%     name = txt(1, 2:end)
%     name_stay = name(stay)
end



% for i = 1:num_SelectedGenes
%     SelectedGenes{ i } = GeneNames{ FeatureRank( i ) };
% end
% SelectedGenes
