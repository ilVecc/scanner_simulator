function write_ply(points, filepath)
    fid = fopen(filepath, 'w');
    fprintf(fid, 'ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n', size(points)(2));
    fwrite(fid, points, 'float');
    fclose(fid)
endfunction
