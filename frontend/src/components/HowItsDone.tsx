import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import StackDiagramModal from "./StackDiagramModal";
import "./HowItsDone.css";

const PARTICLES = 14;

export default function HowItsDone() {
  const [open, setOpen] = useState(false);
  const [bursting, setBursting] = useState(false);

  const handleClick = () => {
    if (open) return;
    setBursting(true);
    setOpen(true);
    // Let the burst animation play; particles clean themselves up.
    window.setTimeout(() => setBursting(false), 650);
  };

  return (
    <>
      <div className="hid">
        <AnimatePresence>
          {bursting && (
            <motion.div
              className="hid__burst"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              key="burst"
            >
              {Array.from({ length: PARTICLES }).map((_, i) => {
                const angle = (i / PARTICLES) * Math.PI * 2;
                const dist = 46 + (i % 3) * 12;
                return (
                  <motion.span
                    key={i}
                    className="hid__particle"
                    initial={{ x: 0, y: 0, scale: 1, opacity: 1 }}
                    animate={{
                      x: Math.cos(angle) * dist,
                      y: Math.sin(angle) * dist,
                      scale: 0,
                      opacity: 0,
                    }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                  />
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>

        <motion.button
          type="button"
          className="hid__pill"
          onClick={handleClick}
          aria-haspopup="dialog"
          aria-expanded={open}
          animate={
            bursting
              ? { scale: [1, 1.25, 0], opacity: [1, 1, 0] }
              : { scale: 1, opacity: open ? 0 : 1, y: [0, -6, 0] }
          }
          transition={
            bursting
              ? { duration: 0.45, ease: "easeIn" }
              : { y: { duration: 4, repeat: Infinity, ease: "easeInOut" }, opacity: { duration: 0.3 } }
          }
          whileHover={{ scale: bursting ? undefined : 1.05 }}
        >
          <span className="hid__pill-glow" aria-hidden="true" />
          <span className="hid__pill-label">How is it done?</span>
        </motion.button>
      </div>

      <StackDiagramModal open={open} onClose={() => setOpen(false)} />
    </>
  );
}
