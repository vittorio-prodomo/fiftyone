import { PillButton } from "@fiftyone/components";
import {
  getSampleSrc,
  getStandardizedUrls,
  modalSample,
  selectedMediaField,
} from "@fiftyone/state";
import { Download as DownloadIcon } from "@mui/icons-material";
import { CircularProgress } from "@mui/material";
import { useCallback, useState } from "react";
import { useRecoilValue } from "recoil";

const Download = () => {
  const [downloading, setDownloading] = useState(false);
  const { sample, urls: rawUrls } = useRecoilValue(modalSample);
  const mediaField = useRecoilValue(selectedMediaField(true));

  const handleDownload = useCallback(() => {
    if (downloading) return;

    const urls = getStandardizedUrls(rawUrls);
    const rawUrl =
      urls?.[mediaField] ?? urls?.filepath ?? (sample?.filepath as string);
    if (!rawUrl) return;

    // getSampleSrc builds "/media?filepath=..." — swap to "/media/download"
    const mediaSrc = getSampleSrc(rawUrl as string);
    const downloadSrc = mediaSrc.replace("/media?", "/media/download?");

    const a = document.createElement("a");
    a.href = downloadSrc;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    setDownloading(true);
    setTimeout(() => setDownloading(false), 2000);
  }, [downloading, rawUrls, mediaField, sample]);

  return (
    <PillButton
      icon={
        downloading ? (
          <CircularProgress size={16} thickness={6} />
        ) : (
          <DownloadIcon />
        )
      }
      open={false}
      highlight={false}
      onClick={handleDownload}
      tooltipPlacement="bottom"
      title="Download media"
      data-cy="action-download"
      style={{ opacity: downloading ? 0.5 : undefined }}
    />
  );
};

export default Download;
