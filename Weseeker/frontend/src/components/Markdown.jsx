import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// 统一的 markdown 渲染：表格、代码块、列表、行内代码、链接
// 样式跟整体 Claude 风格对齐
const components = {
  // 段落：正文行高 1.78
  p: ({ node, ...props }) => (
    <p className="text-[15px] leading-[1.78] text-ink mb-3 last:mb-0" {...props} />
  ),
  // 标题
  h1: ({ node, ...props }) => <h1 className="font-serif text-[20px] mt-4 mb-2" {...props} />,
  h2: ({ node, ...props }) => <h2 className="font-serif text-[17px] mt-3 mb-2" {...props} />,
  h3: ({ node, ...props }) => <h3 className="font-serif text-[15px] mt-3 mb-1.5 font-semibold" {...props} />,
  // 列表
  ul: ({ node, ...props }) => <ul className="list-disc pl-6 space-y-1 mb-3 text-[15px] leading-[1.7]" {...props} />,
  ol: ({ node, ...props }) => <ol className="list-decimal pl-6 space-y-1 mb-3 text-[15px] leading-[1.7]" {...props} />,
  li: ({ node, ...props }) => <li className="text-ink" {...props} />,
  // 行内代码
  code: ({ node, inline, className, children, ...props }) => {
    if (inline) {
      return (
        <code className="font-mono text-[13px] bg-clay-soft text-ink px-1.5 py-0.5 rounded" {...props}>
          {children}
        </code>
      );
    }
    return (
      <code className="font-mono text-[13px] block" {...props}>
        {children}
      </code>
    );
  },
  // 代码块
  pre: ({ node, ...props }) => (
    <pre className="json-block overflow-x-auto mb-3" {...props} />
  ),
  // 链接
  a: ({ node, ...props }) => (
    <a className="text-clay underline underline-offset-2 hover:opacity-80" target="_blank" rel="noreferrer" {...props} />
  ),
  // 引用
  blockquote: ({ node, ...props }) => (
    <blockquote className="border-l-2 border-clay pl-4 my-3 text-ink-2 italic" {...props} />
  ),
  // 表格
  table: ({ node, ...props }) => (
    <div className="overflow-x-auto my-3">
      <table className="border-collapse text-[13px] w-full" {...props} />
    </div>
  ),
  th: ({ node, ...props }) => (
    <th className="border border-line bg-[#fbfaf7] px-3 py-1.5 text-left font-medium" {...props} />
  ),
  td: ({ node, ...props }) => (
    <td className="border border-line px-3 py-1.5" {...props} />
  ),
  // 强调
  strong: ({ node, ...props }) => <strong className="font-semibold text-ink" {...props} />,
  em: ({ node, ...props }) => <em className="italic" {...props} />,
  // 水平线
  hr: () => <hr className="my-4 border-line" />,
};

export default function Markdown({ children }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
