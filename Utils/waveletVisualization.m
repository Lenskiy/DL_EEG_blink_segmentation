layers  = [sequenceInputLayer(5, MinLength=512)
          cwtLayer(SignalLength=512,  FrequencyLimits=[0.25/128 12/128], VoicesPerOctave=48, Wavelet="Morse", TransformMode="mag")];
dlnet   = dlnetwork(layers);

id = 2;
dataout = forward(dlnet, dlarray(trainSet{1}{id}, "CTB"));
q = extractdata(dataout);
q = permute(q,[1 4 2 3]);
figure, hold on;
subplot(2,1,1), plot(trainSet{1}{id}', LineWidth=2);
xlim([0, 512]);
subplot(2,1,2), imagesc(squeeze(q(:,:,id)))