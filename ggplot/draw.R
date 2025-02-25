library("ggplot2")
library("patchwork")
data <- mpg

# 基本要素，data+aesthetic mapping+layer (通常由geom函数创建）
# ggplot(mpg, aes(x = displ, y = hwy, color = class)) +
#   geom_point()

# ggplot(economics, aes(date, unemploy)) + geom_line()

# 分面 facet： grid and wrap
## 分面的话要注意如果不设置scale_x_discrete(y_scale=free)，会导致x轴的值不是子图分开排列
# TODO: scale_x_discrete(y_scale=free)靠记忆写的，要确认
# ggplot(mpg, aes(displ, hwy)) +
#   geom_point() +
#   geom_smooth() +
#   # default method loess apply for <1000 data points
#   # <1000 points, use geom_smooth(method = "gam", formula = y ~ s(x)) provide by mgcv package
#   facet_wrap(~class)

# reorder
p1<-ggplot(mpg, aes(class, hwy)) + geom_boxplot()  # 默认按字母排序
p2<-ggplot(mpg, aes(reorder(class, hwy), hwy)) + geom_boxplot()  # 按hwy值排序
final_p <- p1 + p2
print(final_p)