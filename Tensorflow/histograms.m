close all
acc1 = load('FC_quant_nn_28x28_8b.txt');
acc2 = load('FC_quant_nn_28x28_4b.txt');
acc3 = load('FC_quant_nn_11x11_8b.txt');
acc4 = load('FC_quant_nn_11x11_4b.txt');

str1 = strcat(num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat(num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat(num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat(num2str(median(acc4)),'/',num2str(iqr(acc4)));


red_fill = [255 213 213]/255;

f = figure;

        subplot(2,2,1), histogram(acc1,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        
        %%
        close all
acc1 = load('aconnect_bw_nn_28x28_8b.txt');
acc2 = load('aconnect_bw_nn_28x28_4b.txt');
acc3 = load('aconnect_bw_nn_11x11_8b.txt');
acc4 = load('aconnect_bw_nn_11x11_4b.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


red_fill = [255 213 213]/255;

f = figure;

        subplot(2,2,1), histogram(acc1,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        
        %%

        close all
acc1 = load('aconnect_nn_28x28_8b.txt');
acc2 = load('aconnect_nn_28x28_4b.txt');
acc3 = load('aconnect_nn_11x11_8b.txt');
acc4 = load('aconnect_nn_11x11_4b.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


red_fill = [255 213 213]/255;

f = figure;

        subplot(2,2,1), histogram(acc1,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        
%%
acc1 = load('Derr_aconnect_bw_nn_28x28_8b_50_50.txt');
str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
red_fill = [255 213 213]/255;
f = figure;

        histogram(acc1,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');
%%
acc1 = load('aconnect_bw_nn_28x28_8b_50_50.txt');
str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
red_fill = [255 213 213]/255;
f = figure;

        histogram(acc1,'LineWidth',3,'FaceColor',red_fill), title('MC for MNIST 28x28 8 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

