function pts = read_ply(fn)

    prop_name = {};
    prop_type = {};
    fmt = '';
    header_len = 0;

    fid = fopen(fn);
    line = fgets(fid); % read first line
    header_len = header_len + length(line); % increase header length, includes EOL
    while ~startsWith(line, "end_header")
        line = strsplit(line); % split string

        if startsWith('format', line{1})
            if startsWith('ascii', line{2}) % test whether file is ascii
              fmt = 'ascii';
            elseif startsWith('binary_little_endian', line{2})
              fmt = 'ieee-le';
            else
              fmt = 'ieee-be';
            end
        elseif startsWith('element', line{1}) && startsWith('vertex', line{2}) % number of points
            N = str2num(line{3});
        elseif startsWith('property', line{1}) % identify fields
            prop_type = [prop_type, line{2}];
            prop_name = [prop_name, line{3}];
        end
        line = fgets(fid);
        header_len = header_len + length(line);
    end


    pts = struct();

    if ~strcmp(fmt, 'ascii')
        % total file length minus header
        type_size = struct('int', 4, 'float', 4, 'float64', 8);
        fseek(fid, 0, 1);
        file_size = ftell(fid) - header_len;
        line_size = int32(file_size / N);
        offset = 0;

        for i = 1:length(prop_name)
            fseek(fid, header_len + offset);
            dt = type_size.(prop_type{i}); % prop_type for field
            prec = strcat([prop_type{i},'=>',prop_type{i}]);
            pts.(prop_name{i}) = fread(fid, N, prec, line_size - dt, fmt);
            offset = offset + dt;
        end
    end

    fclose(fid);
end
