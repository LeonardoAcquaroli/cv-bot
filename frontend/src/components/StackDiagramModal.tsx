import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import "./StackDiagramModal.css";

interface Props {
  open: boolean;
  onClose: () => void;
}

interface NodeSpec {
  label: string;
  sub: string;
  icon: string;
}

const FLOW: NodeSpec[] = [
  { label: "You", sub: "ask a question", icon: "💬" },
  { label: "React + TypeScript", sub: "Vite frontend", icon: "⚛️" },
  { label: "FastAPI", sub: "Python backend · SSE", icon: "⚡" },
];

const SERVICES: NodeSpec[] = [
  { label: "Qdrant", sub: "vector search on personal docs", icon: "🔎" },
  { label: "OpenAI GPT-5 nano", sub: "streamed answer", icon: "🧠" },
];

export default function StackDiagramModal({ open, onClose }: Props) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="sdm__backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="How this app is built"
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.25 }}
        >
          <motion.div
            className="sdm__panel"
            onClick={(e) => e.stopPropagation()}
            initial={{ scale: 0.4, opacity: 0, y: 40 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.5, opacity: 0, y: 20 }}
            transition={{ type: "spring", stiffness: 260, damping: 24 }}
          >
            <button className="sdm__close" onClick={onClose} aria-label="Close">
              ✕
            </button>

            <div className="sdm__body">
              <h2 className="sdm__title">How is it done?</h2>
              <p className="sdm__subtitle">
                <strong>A single-process monolith</strong>.
                <br /><strong>One FastAPI app</strong> streams answers to <strong>a React frontend.</strong>
                <br /><br />I mean... It's a chatbot, no need to complicate things.
              </p>

              <div className="sdm__flow">
              {FLOW.map((node, i) => (
                <FlowStep key={node.label} node={node} index={i} showConnector={i < FLOW.length} />
              ))}

              <div className="sdm__branch">
                <span className="sdm__branch-line" />
                <div className="sdm__services">
                  {SERVICES.map((node, i) => (
                    <motion.div
                      key={node.label}
                      className="sdm__node sdm__node--service"
                      initial={{ opacity: 0, x: 24 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + i * 0.12 }}
                    >
                      <span className="sdm__node-icon">{node.icon}</span>
                      <span className="sdm__node-label">{node.label}</span>
                      <span className="sdm__node-sub">{node.sub}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function FlowStep({
  node,
  index,
  showConnector,
}: {
  node: NodeSpec;
  index: number;
  showConnector: boolean;
}) {
  return (
    <div className="sdm__step">
      <motion.div
        className="sdm__node"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 + index * 0.12 }}
      >
        <span className="sdm__node-icon">{node.icon}</span>
        <span className="sdm__node-label">{node.label}</span>
        <span className="sdm__node-sub">{node.sub}</span>
      </motion.div>
      {showConnector && (
        <span className="sdm__connector" aria-hidden="true">
          <span className="sdm__connector-dot" style={{ animationDelay: `${index * 0.3}s` }} />
        </span>
      )}
    </div>
  );
}
