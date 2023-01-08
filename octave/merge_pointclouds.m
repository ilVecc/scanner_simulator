%%
clc
clear all
pkg load io


poses = csv2cell('../output/poses.csv');
[N, ~] = size(poses);

U = [1 0 0; 0 -1 0; 0 0 -1];
points = [];
for i = 2:N
    path = poses{i,6};

    R = reshape([poses{i,12:20}], [3,3])'; % transpose because reshape is column major
    T = [poses{i,21:23}]';
    E = [R T; 0 0 0 1];

    pc = read_ply(path);
    hom = ones(1, 100000);
    pc_points = U*[pc.x pc.y pc.z]'; % from blender_cam coords to cv_cam coords
    pc_points = [pc_points; hom];

    points = [points, inv(E)*pc_points]; % blender_world coords
end

% plotting all points cloud
scatter3(points(1,:), points(2,:), points(3,:),'.')
write_ply(points(1:3, :), 'merged.ply');
