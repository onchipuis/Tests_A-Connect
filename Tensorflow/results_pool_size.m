close all
cd ../../
cd Results/11x11/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_11x11_4b_Wstd_50_pool_1xbatch_BW_simerr_0_0.txt');
acc2 = load('AConnect_11x11_4b_Wstd_50_pool_1xbatch_BW_simerr_30.0_30.0.txt');
acc3 = load('AConnect_11x11_4b_Wstd_50_pool_1xbatch_BW_simerr_50.0_50.0.txt');
acc4 = load('AConnect_11x11_4b_Wstd_50_pool_1xbatch_BW_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        %%
        
close all
cd ../../
cd Results/11x11/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_11x11_4b_Wstd_50_pool_2xbatch_BW_simerr_0_0.txt');
acc2 = load('AConnect_11x11_4b_Wstd_50_pool_2xbatch_BW_simerr_30.0_30.0.txt');
acc3 = load('AConnect_11x11_4b_Wstd_50_pool_2xbatch_BW_simerr_50.0_50.0.txt');
acc4 = load('AConnect_11x11_4b_Wstd_50_pool_2xbatch_BW_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        %%
        close all
        cd ../../
cd Results/11x11/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_11x11_4b_Wstd_50_pool_4xbatch_BW_simerr_0_0.txt');
acc2 = load('AConnect_11x11_4b_Wstd_50_pool_4xbatch_BW_simerr_30.0_30.0.txt');
acc3 = load('AConnect_11x11_4b_Wstd_50_pool_4xbatch_BW_simerr_50.0_50.0.txt');
acc4 = load('AConnect_11x11_4b_Wstd_50_pool_4xbatch_BW_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.5,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        %%
        close all
cd ../../
cd Results/28x28/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_28x28_8b_Wstd_50_pool_1xbatch_simerr_0_0.txt');
acc2 = load('AConnect_28x28_8b_Wstd_50_pool_1xbatch_simerr_30.0_30.0.txt');
acc3 = load('AConnect_28x28_8b_Wstd_50_pool_1xbatch_simerr_50.0_50.0.txt');
acc4 = load('AConnect_28x28_8b_Wstd_50_pool_1xbatch_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        %%
        
close all
cd ../../
cd Results/28x28/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_28x28_8b_Wstd_50_pool_2xbatch_simerr_0_0.txt');
acc2 = load('AConnect_28x28_8b_Wstd_50_pool_2xbatch_simerr_30.0_30.0.txt');
acc3 = load('AConnect_28x28_8b_Wstd_50_pool_2xbatch_simerr_50.0_50.0.txt');
acc4 = load('AConnect_28x28_8b_Wstd_50_pool_2xbatch_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        %%
        close all
        cd ../../
cd Results/28x28/
red_fill = [255 213 213]/255;

acc1 = load('AConnect_28x28_8b_Wstd_50_pool_4xbatch_simerr_0_0.txt');
acc2 = load('AConnect_28x28_8b_Wstd_50_pool_4xbatch_simerr_30.0_30.0.txt');
acc3 = load('AConnect_28x28_8b_Wstd_50_pool_4xbatch_simerr_50.0_50.0.txt');
acc4 = load('AConnect_28x28_8b_Wstd_50_pool_4xbatch_simerr_70.0_70.0.txt');

str1 = strcat('M/IQR:',' ',num2str(median(acc1)),'/',num2str(iqr(acc1)));
str2 = strcat('M/IQR:',' ',num2str(median(acc2)),'/',num2str(iqr(acc2)));
str3 = strcat('M/IQR:',' ',num2str(median(acc3)),'/',num2str(iqr(acc3)));
str4 = strcat('M/IQR:',' ',num2str(median(acc4)),'/',num2str(iqr(acc4)));


f = figure;
        
        subplot(2,2,1), bar(acc1,1,0.2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=0%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.5 0.4 0.4],'String',str1,'FitBoxToText','on');

        subplot(2,2,2), histogram(acc2,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=30%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.5 0.4 0.4],'String',str2,'FitBoxToText','on');

        subplot(2,2,3), histogram(acc3,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=50%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.15 0.03 0.4 0.4],'String',str3,'FitBoxToText','on');

        subplot(2,2,4), histogram(acc4,'LineWidth',2,'FaceColor',red_fill), title('MC for MNIST 11x11 4 bits @Simerr=70%'),
        xlabel('Validation Accuracy'), ylabel('Samples')
        annotation(f,'textbox',[0.59 0.03 0.4 0.4],'String',str4,'FitBoxToText','on');
        
        
        