function SaveGIF (h, filename, mode, mode2)
	frame = getframe(h);
	im = frame2im(frame);
	[imind,cm] = rgb2ind(im,256);
	imwrite(imind,cm,filename,mode,mode2,'DelayTime',0);
end