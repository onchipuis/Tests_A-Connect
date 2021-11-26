clc, clear all, close all
blue_line = [0 0 255]/255;
blue_fill = [215 227 240]/255;
red_line = [255 0 0]/255;
red_fill = [255 213 213]/255;
violet_line = [144 44 90]/255;
violet_fill = [218 134 173]/255;
green_line = [44 189 110]/255;
green_fill = [160 233 193]/255;
gray_line = 128*[1 1 1]/255;
gray_fill = 211*[1 1 1]/255;
%A-Connect 30 = 1
%A-Connect 50 = 2
%A-Connect 70 = 3
Werr = [1 2 4 8 16 32 64 128 256];
for j=1:3
	switch j
		case 1
		%k=1 Simerr=0%
		%k=2 Simerr=30%
		%k=3 Simerr=50%
		%k=4 Simerr=70%
			for k=1:4
				switch k
					case 1
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_30_Bstd_30_simerr_0_0.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end
                		figure(1)
						subplot(8,2,1),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 30% with simerr 0%")
						subplot(8,2,2),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 30% with simerr 0%")
					case 2
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_30_Bstd_30_simerr_30_30.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(1)
						subplot(8,2,3),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 30% with simerr 30%")
						subplot(8,2,4),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 30% with simerr 30%")
					case 3
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_30_Bstd_30_simerr_50_50.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(1)
						subplot(8,2,5),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 30% with simerr 50%")
						subplot(8,2,6),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 30% with simerr 50%")
					case 4
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_30_Bstd_30_simerr_70_70.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(1)
						subplot(8,2,7),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 30% with simerr 70%")
						subplot(8,2,8),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 30% with simerr 70%")

				end
			end

		case 2
			for k=1:4
		%k=1 Simerr=0%
		%k=2 Simerr=30%
		%k=3 Simerr=50%
		%k=4 Simerr=70%
				switch k
					case 1
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_50_Bstd_50_simerr_0_0.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(2)
						subplot(8,2,1),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 50% with simerr 0%")
						subplot(8,2,2),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 50% with simerr 0%")
					case 2
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_50_Bstd_50_simerr_30_30.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(2)
						subplot(8,2,3),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 50% with simerr 30%")
						subplot(8,2,4),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 50% with simerr 30%")
					case 3
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_50_Bstd_50_simerr_50_50.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(2)
						subplot(8,2,5),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 50% with simerr 50%")
						subplot(8,2,6),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 50% with simerr 50%")
					case 4
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_50_Bstd_50_simerr_70_70.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(2)
						subplot(8,2,7),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 50% with simerr 70%")
						subplot(8,2,8),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 50% with simerr 70%")

				end
			end

		case 3
		%k=1 Simerr=0%
		%k=2 Simerr=30%
		%k=3 Simerr=50%
		%k=4 Simerr=70%
			for k=1:4
				switch k
					case 1
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_70_Bstd_70_simerr_0_0.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(3)
						subplot(8,2,1),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 70% with simerr 0%")
						subplot(8,2,2),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 70% with simerr 0%")
					case 2
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_70_Bstd_70_simerr_30_30.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(3)
						subplot(8,2,3),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 70% with simerr 30%")
						subplot(8,2,4),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 70% with simerr 30%")

					case 3
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_70_Bstd_70_simerr_50_50.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(3)
						subplot(8,2,5),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 70% with simerr 50%")
						subplot(8,2,6),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 70% with simerr 50%")
					case 4
						Data = zeros(9,2);
						name1="Stats_AConnect_";
						name2= "Werr_28x28_8bits_Wstd_70_Bstd_70_simerr_70_70.txt";
						for i=1:9
							n = 2^(i-1);
							chr = int2str(n);
							name3 = strcat(name1,chr);
							filename=strcat(name3,name2);
							data = importdata(filename);
							Data(i,1) = data(1);
							Data(i,2) = data(2);
						end

						figure(3)
						subplot(8,2,7),plot(Werr,Data(:,1),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("Median"), title("A-Connect 70% with simerr 70%")
						subplot(8,2,8),plot(Werr,Data(:,2),'LineWidth',2, 'Color', blue_line), xlabel("Number of matrices"), ylabel("IQR"), title("A-Connect 70% with simerr 70%")

				end
			end

    end
end
