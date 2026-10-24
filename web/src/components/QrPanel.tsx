import { QRCodeSVG } from "qrcode.react";

// Judges scan this to open the board on their phone and submit a behavior.
export default function QrPanel() {
  return (
    <figure className="qr">
      <QRCodeSVG
        value={location.origin}
        size={84}
        fgColor="#25372f"
        bgColor="transparent"
        title="QR code linking to this board"
      />
      <figcaption>Scan to try it on your phone</figcaption>
    </figure>
  );
}
