import { QRCodeSVG } from "qrcode.react";

// Judges scan this to open the board on their phone and submit a behavior. Big enough
// to scan from across a table, with the address under it for those who type.
export default function QrPanel() {
  return (
    <figure className="qr">
      <QRCodeSVG
        value={location.origin}
        size={168}
        fgColor="#15140f"
        bgColor="#b6e36b"
        marginSize={2}
        title="QR code linking to this board"
      />
      <figcaption>
        Scan to try it yourself
        <span className="qr-url">{location.host}</span>
      </figcaption>
    </figure>
  );
}
