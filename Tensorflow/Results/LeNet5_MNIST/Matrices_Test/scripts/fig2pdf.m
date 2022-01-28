%% FUNCTIONS
function fig2pdf(fig_name,file_name);
% Function to save a matlab plot to a PDF file:
% fig_name: name of the figure
% file_name: name of the file where will be saved
% How to use it:
%   fig2pdf(<fig_name>,<file_name>)
%
  print(fig_name,'-dpdf',file_name)
  crop_sys = ['pdfcrop',' ',file_name,' ',file_name];
  system(crop_sys);
end
