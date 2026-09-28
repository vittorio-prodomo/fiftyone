import { PillButton } from "@fiftyone/components";
import {
  getSampleSrc,
  useModalMediaPath,
  useModalSample,
} from "@fiftyone/state";
import { Icon, IconName, Size, Spinner } from "@voxel51/voodo";
import { useCallback, useState } from "react";

const Download = () => {
  const [downloading, setDownloading] = useState(false);
  const sample = useModalSample();
  const mediaPath = useModalMediaPath();

  const handleDownload = useCallback(() => {
    if (downloading) return;

    const rawUrl = mediaPath ?? (sample?.sample?.filepath as string);
    if (!rawUrl) return;

    // getSampleSrc builds "/media?filepath=..." — swap to "/media/download"
    const mediaSrc = getSampleSrc(rawUrl);
    const downloadSrc = mediaSrc.replace("/media?", "/media/download?");

    const a = document.createElement("a");
    a.href = downloadSrc;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    setDownloading(true);
    setTimeout(() => setDownloading(false), 2000);
  }, [downloading, mediaPath, sample]);

  return (
    <PillButton
      icon={
        downloading ? (
          <Spinner size={Size.Sm} />
        ) : (
          // Unlike MUI icons, VOODO icons have no intrinsic size; 18px
          // matches the neighboring action bar icons
          <Icon name={IconName.Download} size={18} />
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
