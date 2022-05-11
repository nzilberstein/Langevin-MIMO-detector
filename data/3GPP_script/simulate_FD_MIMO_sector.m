addpath('/Users/nicolasz/Projects/MIMO_detection/QuaDriGa_2020.11.03_v2.4.0/quadriga_src')
%%
%PARAMATERS

% modulation paramaters
fc   =  3.5 * 10^9;  % carrier frequency (Hz)
df   =  30 * 10^3;   % subcarrier spacing (Hz)
Nf   =  273 * 12;    % number of subcarriers
BW   =  Nf * df;     % bandwidth (Hz)
Nf_m =  273 * 12;

% gNB antenna paramaters
nGnbAntHor     = 8;
nGnbAntVert    = 8;
nGnbAnt        = nGnbAntHor * nGnbAntVert;
pol            = 1;
downTilt       = 5;   % electric downtilt (degrees)
antSpacing     = 0.5; % antenna spacing (wavelengths)

% geometry
gNB_pos         = [0 0 25]';
cell_radius     = 500; 
nUes            = 100;


scenerio_string = '3GPP_38.901_UMa_NLOS_O2I';
% '3GPP_38.901_UMa_LOS_O2I', '3GPP_38.901_UMa_NLOS_O2I
% '3GPP_38.901_UMa_NLOS';

%%
%DROP USERS

% first drop users in square:
nDrops        =  10^6;
drop_pos      =  zeros(2,nDrops);
drop_pos(1,:) =  cell_radius * rand(1,nDrops);
drop_pos(2,:) = -cell_radius + 2 * cell_radius * rand(1,nDrops);

% keeps user in 120 degree sector:
[theta,rho]  =  cart2pol(drop_pos(1,:),drop_pos(2,:));
keep_idx     =  (rho >= 10) .* (rho <= cell_radius) .*  (theta <= pi/3) .*  (theta >= -pi/3);
kept_ue_pos  =  drop_pos(:,logical(keep_idx));

% grab the desired number of users:
if(sum(keep_idx) < nUes)
    error('did not generate enough user, increase dropsize');
else
    ue_pos        =  zeros(3,nUes);
    ue_pos(3,:)   =  20*ones(1,nUes);
    ue_pos(1:2,:) =  kept_ue_pos(:,1:nUes);
end

%%
%QUAD PRMS

l                         = qd_layout; 
l.simpar.center_frequency = fc;
l.tx_array                = qd_arrayant('3gpp-3d',nGnbAntVert,nGnbAntHor,fc,pol,downTilt,antSpacing);
l.rx_array                = qd_arrayant('omni');
l.no_rx                   = nUes;
l.rx_position             = ue_pos;
l.tx_position             = gNB_pos;
l.set_scenario(scenerio_string);

%%
%QUAD COEFF

b          = l.init_builder;      
nScenerios = length(b);
c          = cell(10,1);
nUePerScen = zeros(10,1);

for scenIdx = 1 : nScenerios
    b_scen = b(scenIdx);
    
    b_scen.scenpar.PerClusterDS = 1;  
    b_scen.scenpar.SC_lambda = 5;
    b_scen.gen_ssf_parameters; 
    
    nUePerScen(scenIdx) =  size(b_scen.rx_positions,2);
    c{scenIdx}          =  b_scen.get_channels;
end


%%
%CHANNEL BANK

H_bank  =  zeros(nGnbAnt,Nf_m,nUes);

ueIdx = 0;
for scenIdx = 1 : nScenerios
    c_scen = c{scenIdx};
    
    for i = 1 : nUePerScen(scenIdx)
        ueIdx = ueIdx + 1;

        c_ue = c_scen(i);
        H_ue = squeeze(c_ue.fr(BW,Nf_m));
    
        avgE   =  mean(abs(H_ue(:)).^2);
        lambda =  1 / sqrt(avgE);
    
        nTaps             =  length(c_ue.delay);
        H_bank(:,:,ueIdx) =  lambda * H_ue;
    end
end




H_bank = permute(H_bank,[2 1 3]);

save('H_bank.mat','H_bank');




    

































