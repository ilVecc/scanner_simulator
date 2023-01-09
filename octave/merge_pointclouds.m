clc
clear all
pkg load io


csv = csv2cell('../output/poses.csv');
header = csv(1, 2:end);
data = csv(2:end, 1:end);
N = size(data, 1);

U = [1 0 0; 0 -1 0; 0 0 -1];  % from blender_cam coords to cv_cam coords
points = [];
for i = 1:N
    path = data{i,6};

    R = reshape([data{i,12:20}], [3,3])';  % transpose because reshape is column major
    T = [data{i,21:23}]';
    E = [R T; 0 0 0 1];

    pc = read_ply(path);
    cv_points = U * [pc.x pc.y pc.z]';
    homo_ones = ones(1, size(cv_points, 2));
    homo_points = [cv_points; homo_ones];

    points_in_world = E \ homo_points;  % blender_world coords
    filt = sum(points_in_world < 1000000, 1) == 4;
    points_in_world = points_in_world(:, filt);
    points = [points, points_in_world];
end
clear path i pc cv_points homo_ones homo_points points_in_world filt

% plotting all points cloud
scatter3(points(1,:), points(2,:), points(3,:),'.');
write_ply(points(1:3, :), 'merged.ply');

