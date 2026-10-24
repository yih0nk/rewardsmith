import { QRCodeSVG } from "qrcode.react";

// Judges scan this to open the board on their phone and submit a behavior.
export default function QrPanel() {
  return (
    <figure className="qr">
      <QRCodeSVG
        value={location.origin}
        size={96}
        fgColor="#000000"
        bgColor="#66ff66"
        marginSize={2}
        title="QR code linking to this board"
      />
      <figcaption>Scan to try it yourself</figcaption>
    </figure>
  );
}
