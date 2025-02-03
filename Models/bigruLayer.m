classdef bigruLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % Example custom residual block layer.


    properties (Learnable, State)
        % Nested dlnetwork objects with both learnable
        % parameters and state parameters.
    
        % Residual block.
        Network
    end
    
    methods
        function layer = bigruLayer(numHiddenUnits, NameValueArgs)
            % layer = bigruLayer(numFilters, Name=Value, OutputMode) specifies
            % additional options using one or more name-value arguments:
            % 
            %     numHiddenUnits          
            %
            %     OutputMode 
            %
            %     Name                   - Layer name
            %                              (default '')
        
            % Parse input arguments.
            arguments
                numHiddenUnits
                NameValueArgs.Name = ''
                NameValueArgs.OutputMode = 'sequence'
            end
        
            outputMode = NameValueArgs.OutputMode;
            name = NameValueArgs.Name;
        
            % Set layer name.
            layer.Name = name;
        
            % Set layer description.
            description = "BiGRU layer with " + numHiddenUnits + " units";
            layer.Description = description;
            
            % Set layer type.
            layer.Type = "GRU layer";

            % Define nested layer graph.
            layers = [functionLayer(@(x) x, Name="input")
                      gruLayer(numHiddenUnits, OutputMode=NameValueArgs.OutputMode, Name="gru_1")
                      concatenationLayer(1, 2, Name="cat")
                      ];
            
            lgraph = layerGraph(layers);

            layers = [FlipLayer(Name="flip_1")
                      gruLayer(numHiddenUnits, OutputMode=NameValueArgs.OutputMode, Name="gru_2")
                      FlipLayer(Name="flip_2")];
    
            lgraph = addLayers(lgraph, layers);

            lgraph = connectLayers(lgraph, "input", "flip_1");
            lgraph = connectLayers(lgraph, "flip_2", "cat" + "/in2"); 
            
 
            % Convert to dlnetwork.
            net = dlnetwork(lgraph,Initialize=false);
        
            % Set Network property.
            layer.Network = net;
        end

        function [Z,state] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result and state.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data
            % Outputs:
            %         Z     - Output of layer forward function
            %         state - Layer state

            % Predict using network.
            net = layer.Network;
            [Z,state] = predict(net,X);
            
        end
    end
end


% function layers = bigruLayer(numHiddenUnits, args)
% arguments
%     numHiddenUnits
%     args.Name
%     args.OutputMode = "sequence"
% end
%     layers = [FlipLayer(args.Name)
%               gruLayer(numHiddenUnits, Name=args.Name + "gru", OutputMode=args.OutputMode)
%               FlipLayer(args.Name)];
% 
%     layers = assembleNetwork(layers);
% end