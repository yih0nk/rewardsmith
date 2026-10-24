import { QRCodeSVG } from "qrcode.react";

// Judges scan this to open the board on their phone and submit a behavior.
export default function QrPanel() {
  return (
    <figure className="qr">
      <QRCodeSVG
        value={location.origin}
        size={88}
        fgColor="#1b1d18"
        bgColor="transparent"
        title="QR code linking to this board"
      />
      <figcaption>
        <span className="label">On your phone</span>
        Scan to try it yourself
      </figcaption>
    </figure>
  );
}
