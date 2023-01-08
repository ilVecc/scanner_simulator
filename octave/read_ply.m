function pts = read_ply(fn)

    fid = fopen(fn);
    line = fgetl(fid); % read first line

    len = 0;
    prop = {};
    % dtype_map = {'float': 'f4', 'uchar': 'B', 'int':'i'}
    dtype = {};
    fmt = 'binary';

    while ~strcmp(line, "end_header")
        len = len + length(line) + 1; % increase header length, +1 includes EOL
        line = strsplit(line); % split string

        if strcmp('format', line{1}) && strcmp('ascii', line{2}) % test whether file is ascii
            fmt = 'ascii';
        elseif strcmp('element', line{1}) && strcmp('vertex', line{2}) % number of points
            N = str2num(line{3});
        elseif strcmp('property', line{1}) % identify fields
            dtype = [dtype, line{2}];
            prop = [prop, line{3}];
        endif
        line = fgetl(fid);
    endwhile

    len = len + length(line) + 1; % add 'end_header

    pts = struct();

    if strcmp(fmt, 'binary')
        % total file length minus header
        types = struct('int', 4, 'float', 4, 'float64', 8);
        fseek(fid, 0, 1);
        file_length = ftell(fid) - len;
        seek_plus = 0;

        for i = 1:length(prop)
            fseek(fid, len + seek_plus);
            dt = types.(dtype{i}); % dtype for field
            pts.(prop{i}) = fread(fid, N, dtype{i}, int32(file_length / N) - dt);
            seek_plus = seek_plus + dt;
        endfor
    endif

    fclose(fid);
endfunction
