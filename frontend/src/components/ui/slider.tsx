import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps {
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  label?: string
}

const Slider = React.forwardRef<HTMLInputElement, SliderProps>(
  ({ value, min, max, step, onChange, label }, ref) => {
    return (
      <div className="w-full space-y-2">
        {label && (
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">{label}</span>
            <span className="font-medium">{value.toFixed(1)}</span>
          </div>
        )}
        <input
          ref={ref}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className={cn(
            "w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
          )}
        />
      </div>
    )
  }
)
Slider.displayName = "Slider"

export { Slider }
