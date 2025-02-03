function results = gridSearch2(model, trainSet, testSet, parameters, options, iterateOver)
    rng(1);
    listOfParamters = cartesianProduct(parameters, iterateOver);

    results(length(listOfParamters)) = struct('parameters', [], 'accuracy', [], 'numberOfWeights', []);
    for k = length(listOfParamters):-1:1       % Filter size search
        try
            layers = model(listOfParamters(k));
            tic
            [net, info] = trainNetwork(trainSet{1}, trainSet{2}, layers, options);
            t = toc;
            numberOfWeights = computeNumberOfWeights(net);
            Y_pred = classify(net, testSet{1});
            %%%%%%%%%%%%%%%%%
            valAccuracy = 0;
            for l = 1:length(Y_pred)
                valAccuracy = valAccuracy + mean(Y_pred{l} == testSet{2}{l});
            end

            valAccuracy = 100*valAccuracy/length(Y_pred);
            %%%%%%%%%%%%%%%%%%
            results(k) =  struct('parameters', listOfParamters(k), 'accuracy', valAccuracy, 'numberOfWeights', numberOfWeights);
            
            disp([table(k, [valAccuracy, info.FinalValidationAccuracy], numberOfWeights, t/60, VariableNames = ["k", "accuracy", "numberOfWeights", "time"]), struct2table(listOfParamters(k))])
            
            save(fullfile("Results", string(char(model)) + "_temp.mat"), 'results');
        catch EM
            disp(EM)
        end
    end
end

function numberOfWeights = computeNumberOfWeights(lgraph)
    numberOfWeights = 0;
    for k = 1:numel(lgraph.Layers)
        if isprop(lgraph.Layers(k), "Weights")
            numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).Weights);
        end
        if isprop(lgraph.Layers(k), "Bias")
            numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).Bias);
        end
        if isprop(lgraph.Layers(k), "InputWeights")
            numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).InputWeights);
        end
        if isprop(lgraph.Layers(k), "RecurrentWeights")
            numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).RecurrentWeights);
        end
        if isprop(lgraph.Layers(k), "PeepholeWeights")
            numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).PeepholeWeights);
        end
    end
end