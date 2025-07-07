function setup_parpool()
    pool = gcp('nocreate');
    if isempty(pool)
        available_cores = feature('numcores');
        usable_cores = max(1, available_cores - 4);
        parpool(usable_cores);
    end
end