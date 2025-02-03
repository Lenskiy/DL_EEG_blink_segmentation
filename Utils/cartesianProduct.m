function splittedParamters = cartesianProduct(parameters, iterateOver)
    
    n = length(iterateOver);

    for k = 1:n
        searchedParameters{k} = parameters.(iterateOver(k));
    end

    [F{1:n}] = ndgrid(searchedParameters{:});

    for k=n:-1:1
        G{:,k} = F{k}(:);
    end

    for k = 1:length(G{1})
        splittedParamters(k) = parameters;
        for l = 1:n
            splittedParamters(k).(iterateOver(l)) = G{l}(k);
        end        
    end
end